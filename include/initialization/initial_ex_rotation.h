#ifndef SIMPLE_VIO_INITIAL_EX_ROTATION_H
#define SIMPLE_VIO_INITIAL_EX_ROTATION_H

#include <vector>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace Eigen;

/**
 * 利用旋转约束估计外参数旋转qbc, 相机到IMU的旋转
 */
class InitialEXRotation {
public:
    InitialEXRotation();
    bool calibrationExRotation(vector<pair<Vector3d, Vector3d>>& corres,
                               Quaterniond& delta_q_imu, Matrix3d& calib_ric);
private:
    Matrix3d solveRelativeR(const vector<pair<Vector3d, Vector3d>>& corres);
    double testTriangulation(const vector<cv::Point2f>& l,
                             const vector<cv::Point2f>& r,
                             cv::Mat_<double>& R, cv::Mat_<double>& t);
    void decomposeE(cv::Mat& E,
                    cv::Mat_<double>& R1, cv::Mat_<double>& R2,
                    cv::Mat_<double>& t1, cv::Mat_<double>& t2);
    int frame_count;

    vector<Matrix3d> Rc;
    vector<Matrix3d> Rimu;
    vector<Matrix3d> Rc_g;
    Matrix3d ric;
};

#endif //SIMPLE_VIO_INITIAL_EX_ROTATION_H
