#ifndef SIMPLE_VIO_PARAMETERS_H
#define SIMPLE_VIO_PARAMETERS_H

#include <Eigen/Dense>
/*
 * 通过引用此头文件,能够从外部获取extern参数变量
 */
const int NUM_OF_CAM = 1;       // 相机个数
extern int FOCAL_LENGTH;        // 焦距
extern std::string IMAGE_TOPIC; // 图像处理主题
extern std::string IMU_TOPIC;   // IMU处理主题
extern std::string FISHEYE_MASK;//
extern std::vector<std::string> CAM_NAMES; // 相机名称
extern int MAX_CNT;             //
extern int MIN_DIST;
// extern int WINDOW_SIZE;
extern int FREQ;
extern double F_THRESHOLD;
extern int SHOW_TRACK;
extern bool STEREO_TRACK;
extern int EQUALIZE; //如果光太亮或太暗则为1，进行直方图均衡化
extern int FISHEYE;
extern bool PUB_THIS_FRAME;

//estimator
// const double FOCAL_LENGTH = 460.0;
const int WINDOW_SIZE = 10;
// const int NUM_OF_CAM = 1;
const int NUM_OF_F = 1000;
//#define UNIT_SPHERE_ERROR

extern double INIT_DEPTH;
extern double MIN_PARALLAX;
extern int ESTIMATE_EXTRINSIC;              // 2:没有任何估计,1:有个初始的估计,0:有准确估计

extern double ACC_N, ACC_W;
extern double GYR_N, GYR_W;

extern std::vector<Eigen::Matrix3d> RIC;    // 相机到IMU旋转
extern std::vector<Eigen::Vector3d> TIC;    // 相机到IMU的平移
extern Eigen::Vector3d G;

extern double BIAS_ACC_THRESHOLD;
extern double BIAS_GYR_THRESHOLD;
extern double SOLVER_TIME;
extern int NUM_ITERATIONS;
extern std::string EX_CALIB_RESULT_PATH;
extern std::string VINS_RESULT_PATH;
extern std::string IMU_TOPIC;
extern double TD;   // 信息读取时间延迟
extern double TR;
extern int ESTIMATE_TD; // 相机和IMU在线估计延迟
extern int ROLLING_SHUTTER;
extern double ROW, COL;

enum SIZE_PARAMETERIZATION
{
    SIZE_POSE = 7,
    SIZE_SPEEDBIAS = 9,
    SIZE_FEATURE = 1
};

enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

enum NoiseOrder
{
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};

void readParameters(const std::string& config_file);

#endif //SIMPLE_VIO_PARAMETERS_H
