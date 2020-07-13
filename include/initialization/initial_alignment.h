#ifndef SIMPLE_VIO_INITIAL_ALIGNMENT_H
#define SIMPLE_VIO_INITIAL_ALIGNMENT_H

#include <Eigen/Dense>
#include <map>

#include "utility/utility.h"
#include "factor/integration_base.h"

using namespace Eigen;
using namespace std;

/**
 * 图像帧的相关操作
 * 包括:特征点,时间戳,位姿R,t, 预积分对象, 是否为关键帧
 */
class ImageFrame {
public:
    ImageFrame() {}
    ImageFrame(const map<int, vector<pair<int, Matrix<double, 7, 1>>>>& _points,
                double _t) : t{-t}, is_key_frame{false}, points{_points} {

    }

    map<int, vector<pair<int, Matrix<double, 7, 1>>>> points;
    double t;
    Matrix3d R;
    Vector3d T;
    IntegrationBase* pre_integration;
    bool is_key_frame;

};

// 对齐视觉和IMU
bool visualIMUAlignment(map<double, ImageFrame>& all_image_frame, Vector3d* bgs,
                        Vector3d& g, VectorXd& x);

#endif //SIMPLE_VIO_INITIAL_ALIGNMENT_H
