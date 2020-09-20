#ifndef SIMPLE_VIO_ESTIMATOR_H
#define SIMPLE_VIO_ESTIMATOR_H

#include "backend/eigen_types.h"
#include "factor/integration_base.h"
#include "parameters.h"
#include "feature_manager.h"
#include "initialization/solve_5pts.h"
#include "initialization/initial_alignment.h"
#include "initialization/initial_ex_rotation.h"

/**
 * 状态估计器
 */
class Estimator {
public:
    Estimator();
    void setParameter();

    // 接口
    void processIMU(double t, const Eigen::Vector3d& linear_acceleration,
                    const Eigen::Vector3d& angular_velocity);
    void processImage(
            const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>& image,
            double header);
    void setReloFrame(double _frame_stamp, int _frame_index,
                      std::vector<Eigen::Vector3d>& _match_points,
                      Eigen::Vector3d _relo_t, Eigen::Matrix3d _relo_r);

    // internal
    void clearState();
    bool initialStructure();
    bool visualInitialAlign();
    bool relativePose(Eigen::Matrix3d& relative_R, Eigen::Vector3d& relative_T, int& l);
    void slideWindow();
    void solveOdometry();
    void slideWindowNew();
    void slideWindowOld();
    void optimization();
    void backendOptimization();

    void problemSolve();
    void margOldFrame();
    void margNewFrame();

    void vector2double();
    void double2vector();
    bool failureDetection();

    enum SolverFlag {
        INITIAL,
        NON_LINEAR
    };

    enum MarginalizationFlag {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };
//////////////// OUR SOLVER ///////////////////
    MatXX Hprior_;
    VecX bprior_;
    VecX errprior_;
    MatXX Jprior_inv_;
    Eigen::Matrix2d project_sqrt_info_;
//////////////// OUR SOLVER //////////////////
    SolverFlag solver_flag;
    MarginalizationFlag  marginalization_flag;
    Eigen::Vector3d g;
    Eigen::MatrixXd Ap[2], backup_A;
    Eigen::VectorXd bp[2], backup_b;

    Eigen::Matrix3d ric[NUM_OF_CAM];
    Eigen::Vector3d tic[NUM_OF_CAM];

    Eigen::Vector3d Ps[(WINDOW_SIZE + 1)];
    Eigen::Vector3d Vs[(WINDOW_SIZE + 1)];
    Eigen::Matrix3d Rs[(WINDOW_SIZE + 1)];
    Eigen::Vector3d Bas[(WINDOW_SIZE + 1)];
    Eigen::Vector3d Bgs[(WINDOW_SIZE + 1)];
    double td;

    Eigen::Matrix3d back_R0, last_R, last_R0;
    Eigen::Vector3d back_P0, last_P, last_P0;
    double Headers[(WINDOW_SIZE + 1)];

    IntegrationBase* pre_integrations[(WINDOW_SIZE + 1)];
    Eigen::Vector3d acc_0, gyr_0;

    std::vector<double> dt_buf[(WINDOW_SIZE + 1)];
    std::vector<Eigen::Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
    std::vector<Eigen::Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];

    int frame_count;
    int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;

    FeatureManager f_manager;
    MotionEstimator m_estimator;
    InitialEXRotation initial_ex_rotation;

    bool first_imu;
    bool is_valid, is_key;
    bool failure_occur;

    std::vector<Eigen::Vector3d> point_cloud;
    std::vector<Eigen::Vector3d> margin_cloud;
    std::vector<Eigen::Vector3d> key_poses;
    double initial_timestamp;

    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
    double para_Feature[NUM_OF_F][SIZE_FEATURE];
    double para_Ex_Pose[NUM_OF_CAM][SIZE_POSE];
    double para_Retrive_Pose[SIZE_POSE];
    double para_Td[1][1];
    double para_Tr[1][1];

    int loop_window_index;

    // MarginalizationInfo *last_marginalization_info;
    std::vector<double*> last_marginalization_parameter_blocks;

    std::map<double, ImageFrame> all_image_frame;
    IntegrationBase* tmp_pre_integration;

    // relocalization variable
    bool relocalization_info;
    double relo_frame_stamp;
    double relo_frame_index;
    int relo_frame_local_index;
    std::vector<Eigen::Vector3d> match_points;
    double relo_Pose[SIZE_POSE];
    Eigen::Matrix3d drift_correct_r;
    Eigen::Vector3d drift_correct_t;
    Eigen::Vector3d prev_relo_t;
    Eigen::Matrix3d prev_relo_r;
    Eigen::Vector3d relo_relative_t;
    Eigen::Quaterniond relo_relative_q;
    double relo_relative_yaw;

    // 记录所有frame和hessian的处理时间
    double total_hessian_time = 0.0;
    // 记录所有frame处理时间
    double total_frame_time = 0.0;
    // 记录所有frame个数
    long total_frame_num = 0;
    // 记录所有frame的solve次数
    long solve_count_per_frame = 0;
};


#endif //SIMPLE_VIO_ESTIMATOR_H
