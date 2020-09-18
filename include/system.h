#ifndef SIMPLE_VIO_SYSTEM_H
#define SIMPLE_VIO_SYSTEM_H

#include "estimator.h"
#include <condition_variable>
#include "feature_tracker.h"
#include "parameters.h"
#include <pangolin/pangolin.h>
#include <Eigen/Dense>
#include <memory>
#include <opencv2/opencv.hpp>

/// IMU信息结构
struct IMU_MSG {
    double header;                          // 时间戳
    Eigen::Vector3d linear_acceleration;    // 加速度信息
    Eigen::Vector3d angular_velocity;       // 角速度信息
};

/// Imu信息的常值指针
typedef std::shared_ptr<IMU_MSG const> ImuConstPtr;

/// 图像信息结构
struct IMG_MSG {
    double header;                          // 时间戳
    std::vector<Eigen::Vector3d> points;    // points
    std::vector<int> id_of_point;           // point-id
    std::vector<float> u_of_point;          // x-u值
    std::vector<float> v_of_point;          // y-v值
    std::vector<float> velocity_x_of_point; // vel_x
    std::vector<float> velocity_y_of_point; // vel_y
};

/// 图像信息的常值指针
typedef std::shared_ptr <IMG_MSG const> ImgConstPtr;

/// vio主系统类
class System {
public:
    System(const std::string& sConfig_files);
    ~System();

    /// 发布图像数据线程
    void pubImageData(double dStampSec, cv::Mat& img);
    /// 发布Imu数据线程
    void pubImuData(double dStampSec, const Eigen::Vector3d& vGyr,
                    const Eigen::Vector3d& vAcc);

    /// visual-imu 后端处理线程
    void processBackEnd();
    /// 系统处理显示线程
    void draw();

    pangolin::OpenGlRenderState s_cam;
    pangolin::View d_cam;

private:
    /// feature tracker
    std::vector<uchar> r_status;
    std::vector<float> r_err;
    // std::queue<ImageConstPtr> img_buf;
    /// 特征跟踪
    FeatureTracker trackerData[NUM_OF_CAM];
    double first_image_time;
    int pub_count = 1;
    bool first_image_flag = true;
    double last_image_time = 0;
    bool init_pub = 0;

    /// estimator
    Estimator estimator;

    std::condition_variable con;
    double current_time = -1;
    std::queue<ImuConstPtr> imu_buf;
    std::queue<ImgConstPtr> feature_buf;
    /// std::queue<PointCloudConstPtr> relo_buf;
    int sum_of_wait = 0;

    std::mutex m_buf;
    std::mutex m_state;
    std::mutex i_buf;
    std::mutex m_estimator;

    double latest_time;
    Eigen::Vector3d tmp_P;
    Eigen::Quaterniond tmp_Q;
    Eigen::Vector3d tmp_V;
    Eigen::Vector3d tmp_Ba;
    Eigen::Vector3d tmp_Bg;
    Eigen::Vector3d acc_0;
    Eigen::Vector3d gyr_0;
    bool init_feature = 0;
    bool init_imu = 1;
    double last_imu_t = 0;
    std::ofstream ofs_pose;
    std::vector<Eigen::Vector3d> vPath_to_draw;
    bool bStart_backend;
    std::vector<std::pair<std::vector<ImuConstPtr>, ImgConstPtr>> getMeasurements();
};

#endif //SIMPLE_VIO_SYSTEM_H
