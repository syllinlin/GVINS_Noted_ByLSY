#include "gnss_vi_initializer.h"

GNSSVIInitializer::GNSSVIInitializer(const std::vector<std::vector<ObsPtr>> &gnss_meas_buf_, 
    const std::vector<std::vector<EphemBasePtr>> &gnss_ephem_buf_, const std::vector<double> &iono_params_)
        : gnss_meas_buf(gnss_meas_buf_), gnss_ephem_buf(gnss_ephem_buf_), iono_params(iono_params_)
{
    num_all_meas = 0;
    all_sat_states.clear();
    for (uint32_t i = 0; i < gnss_meas_buf.size(); ++i)
    {
        num_all_meas += gnss_meas_buf[i].size();
        all_sat_states.push_back(sat_states(gnss_meas_buf[i], gnss_ephem_buf[i]));
    }
}

/**
 * @brief 粗糙定位
 * /
 * 论文里的描述：在没有任何先验信息的情况下，将近期是有伪距测量值作为GNSS SPP算法的输入，得到接收器的一个粗糙ECEF位置
 * 以及四个导航系统中卫星的时钟偏差（钟差）
 * /
 * **/
bool GNSSVIInitializer::coarse_localization(Eigen::Matrix<double, 7, 1> &result)
{
    result.setZero();
    std::vector<ObsPtr> accum_obs;
    std::vector<EphemBasePtr> accum_ephems;
    // 将BUF中所有的卫星观测数据和星历信息分别复制到accum_obs、accum_ephems中
    for (uint32_t i = 0; i < gnss_meas_buf.size(); ++i)
    {
        std::copy(gnss_meas_buf[i].begin(), gnss_meas_buf[i].end(), std::back_inserter(accum_obs));
        std::copy(gnss_ephem_buf[i].begin(), gnss_ephem_buf[i].end(), std::back_inserter(accum_ephems));
    }
    // 伪距定位得到接收机在ECEF系上的位置坐标以及四个卫星时钟的时钟偏差
    Eigen::Matrix<double, 7, 1> xyzt = psr_pos(accum_obs, accum_ephems, iono_params);
    if (xyzt.topLeftCorner<3, 1>().norm() == 0)
    {
        std::cerr << "Failed to obtain a rough reference location.\n";
        return false;
    }

    // 如果相应卫星时钟偏差<1，认为没有观测到该导航系统的卫星，那么将该卫星的时钟偏差置为0
    for (uint32_t k = 0; k < 4; ++k)
    {
        if (fabs(xyzt(k+3)) < 1)
            xyzt(k+3) = 0;          // not observed yet
    }

    result = xyzt;
    return true;
}

/**
 * GNSS和VIO的偏航角对齐：即ENU系和Local world系的Z轴是完全重合的，仅需要知道两系之间的偏航角，就可以完成两系的对齐
 * */ 
bool GNSSVIInitializer::yaw_alignment(const std::vector<Eigen::Vector3d> &local_vs, 
    const Eigen::Vector3d &rough_anchor_ecef, double &aligned_yaw, double &rcv_ddt)
{
    aligned_yaw = 0;
    rcv_ddt = 0;

    // 估计值
    double est_yaw = 0;
    double est_rcv_ddt = 0;

    Eigen::Matrix3d rough_R_ecef_enu = ecef2rotation(rough_anchor_ecef); // ENU ——> ECEF的转换
    uint32_t align_iter = 0;
    double align_dx_norm = 1.0;
    // 最小二乘求解接收机时钟钟差变化率和偏航角
    while (align_iter < MAX_ITERATION && align_dx_norm > CONVERGENCE_EPSILON)
    {
        Eigen::MatrixXd align_G(num_all_meas, 2); // num_all_meas：数据总数量，2：2个优化量
        align_G.setZero();
        align_G.col(1).setOnes(); // 这样设置的话，将接收机钟差变化率作为第二个优化量
        Eigen::VectorXd align_b(num_all_meas);
        align_b.setZero();
        Eigen::Matrix3d align_R_enu_local(Eigen::AngleAxisd(est_yaw, Eigen::Vector3d::UnitZ())); // local——>ENU的转换
        // 绕Z轴旋转est_yaw角度的旋转矩阵的导数
        Eigen::Matrix3d align_tmp_M;
        align_tmp_M << -sin(est_yaw), -cos(est_yaw), 0,
                        cos(est_yaw), -sin(est_yaw), 0,
                        0       , 0        , 0;
        
        // 构建多普勒测量的残差和雅可比矩阵，用于最小二乘求导
        uint32_t align_counter = 0;
        for (uint32_t i = 0; i < gnss_meas_buf.size(); ++i)
        {
            Eigen::Matrix<double, 4, 1> ecef_vel_ddt; // 记录VIO速度转到ECEF下的数值和接收机钟差变化率
            ecef_vel_ddt.head<3>() = rough_R_ecef_enu * align_R_enu_local * local_vs[i]; // 将VIO速度从local 转到 ECEF下
            ecef_vel_ddt(3) = est_rcv_ddt; // 接收机时钟钟差变化率
            Eigen::VectorXd epoch_res; // 多普勒测量残差
            Eigen::MatrixXd epoch_J; // 多普勒测量雅可比
            dopp_res(ecef_vel_ddt, rough_anchor_ecef, gnss_meas_buf[i], all_sat_states[i], epoch_res, epoch_J);
            align_b.segment(align_counter, gnss_meas_buf[i].size()) = epoch_res;
            align_G.block(align_counter, 0, gnss_meas_buf[i].size(), 1) = 
                epoch_J.leftCols(3)*rough_R_ecef_enu*align_tmp_M*local_vs[i]; // 因为dopp_res中仅仅对速度求到，所以需要乘上速度到偏航角的导数
            align_counter += gnss_meas_buf[i].size();
        }
        // 最小二乘求解
        Eigen::VectorXd dx = -(align_G.transpose()*align_G).inverse() * align_G.transpose() * align_b;
        est_yaw += dx(0);
        est_rcv_ddt += dx(1);
        align_dx_norm = dx.norm();
        ++ align_iter;
    }

    if (align_iter > MAX_ITERATION)
    {
        std::cerr << "Fail to initialize yaw offset.\n";
        return false;
    }

    // 规范偏航角，使得偏航角的范围在0～180度之间
    aligned_yaw = est_yaw;
    if (aligned_yaw > M_PI)
        aligned_yaw -= floor(est_yaw/(2.0*M_PI) + 0.5) * (2.0*M_PI);
    else if (aligned_yaw < -M_PI)
        aligned_yaw -=  ceil(est_yaw/(2.0*M_PI) - 0.5) * (2.0*M_PI);

    rcv_ddt = est_rcv_ddt;

    return true;
}

/***
 * @brief 对anchor point位置精细化，并求出接收机的时钟钟差dt
 * @param[in] local_ps VIO求解出相机在local world下的位置（已求解）
 * @param[in] aligned_yaw ENU和local world之间的偏航角（已求解）
 * @param[in] aligned_ddt 接收机的时钟种差变化率（已求解）
 * @param[in] rough_ecef_dt 接收机在ECEF下的粗糙位置以及四个导航系统的卫星时钟钟差（已求解）
 * @param[out] refined_ecef_dt 再优化后的接收机的ECEF位置和接收机在四个导航系统下不同的钟差
*/
bool GNSSVIInitializer::anchor_refinement(const std::vector<Eigen::Vector3d> &local_ps, 
    const double aligned_yaw, const double aligned_ddt, 
    const Eigen::Matrix<double, 7, 1> &rough_ecef_dt, Eigen::Matrix<double, 7, 1> &refined_ecef_dt)
{
    refined_ecef_dt.setZero(); // 状态清空，置零

    // 轴角构建local——ENU的旋转矩阵
    Eigen::Matrix3d aligned_R_enu_local(Eigen::AngleAxisd(aligned_yaw, Eigen::Vector3d::UnitZ()));

    // refine anchor point and receiver clock bias
    Eigen::Vector3d refine_anchor = rough_ecef_dt.head<3>(); // 用第一步得到的伪距位置作为初值
    Eigen::Vector4d refine_dt = rough_ecef_dt.tail<4>();
    uint32_t refine_iter = 0;
    double refine_dx_norm = 1.0;
    std::vector<uint32_t> unobserved_sys; // 记录没有被观测到的导航系统
    for (uint32_t k = 0; k < 4; ++k)
    {
        if (rough_ecef_dt(3+k) == 0)
            unobserved_sys.push_back(k);
    }

    // 构建相应残差和雅可比矩阵，进行变量的最小二乘求解
    while (refine_iter < MAX_ITERATION && refine_dx_norm > CONVERGENCE_EPSILON)
    {
        Eigen::MatrixXd refine_G(num_all_meas+unobserved_sys.size(), 7);
        Eigen::VectorXd refine_b(num_all_meas+unobserved_sys.size());
        refine_G.setZero();
        refine_b.setZero();
        uint32_t refine_counter = 0;
        Eigen::Matrix3d refine_R_ecef_enu = ecef2rotation(refine_anchor); 
        Eigen::Matrix3d refine_R_ecef_local = refine_R_ecef_enu * aligned_R_enu_local;
        for (uint32_t i = 0; i < gnss_meas_buf.size(); ++i)
        {
            Eigen::Matrix<double, 7, 1> ecef_xyz_dt;
            ecef_xyz_dt.head<3>() = refine_R_ecef_local * local_ps[i] + refine_anchor; // 论文中公式（14）
            ecef_xyz_dt.tail<4>() = refine_dt + aligned_ddt * i * Eigen::Vector4d::Ones(); //  论文中公式（23）

            Eigen::VectorXd epoch_res;
            Eigen::MatrixXd epoch_J;
            std::vector<Eigen::Vector2d> tmp_atmos_delay, tmp_sv_azel;
            psr_res(ecef_xyz_dt, gnss_meas_buf[i], all_sat_states[i], iono_params, 
                epoch_res, epoch_J, tmp_atmos_delay, tmp_sv_azel); // 对Panc构建残差和雅可比矩阵
            refine_b.segment(refine_counter, gnss_meas_buf[i].size()) = epoch_res;
            refine_G.middleRows(refine_counter, gnss_meas_buf[i].size()) = epoch_J;
            refine_counter += gnss_meas_buf[i].size();
        }
        // 对接收机钟差构建残差和雅可比矩阵
        for (uint32_t k : unobserved_sys)
        {
            refine_b(refine_counter) = 0;
            refine_G(refine_counter, k+3) = 1.0;
            ++ refine_counter;
        }

        // 最小二乘求解
        Eigen::VectorXd dx = -(refine_G.transpose()*refine_G).inverse() * refine_G.transpose() * refine_b;
        refine_anchor += dx.head<3>();
        refine_dt += dx.tail<4>();
        refine_dx_norm = dx.norm();
        ++ refine_iter;
    }

    if (refine_iter > MAX_ITERATION)
    {
        std::cerr << "Fail to perform anchor refinement.\n";
        return false;
    }

    refined_ecef_dt.head<3>() = refine_anchor;
    refined_ecef_dt.tail<4>() = refine_dt;

    return true;
}