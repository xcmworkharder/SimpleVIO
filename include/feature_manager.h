#ifndef SIMPLE_VIO_FEATURE_MANAGER_H
#define SIMPLE_VIO_FEATURE_MANAGER_H

#include <Eigen/Dense>
#include <list>
#include <map>
#include "parameters.h"

/**
 * 单帧特征类:单帧图像上某个特征对应的详细信息
 */
class FeaturePerFrame {
public:
    FeaturePerFrame(const Eigen::Matrix<double, 7, 1>& _point, double td) {
        point.x() = _point(0);
        point.y() = _point(1);
        point.z() = _point(2);
        uv.x() = _point(3);
        uv.y() = _point(4);
        velocity.x() = _point(5);
        velocity.y() = _point(6);
        cur_td = td;
    }

    Eigen::Vector3d point;
    Eigen::Vector2d uv;
    Eigen::Vector2d velocity;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    double cur_td;
    double z;
    bool is_used;
    double parallax;
    double dep_gradient;
};

/**
 * 索引为Id的特征点对应的所有关联帧信息
 */
class FeaturePerId {
public:
    FeaturePerId(int _feature_id, int _start_frame)
            : feature_id(_feature_id), start_frame(_start_frame),
              used_num(0), estimated_depth(-1.0), solve_flag(0) {
    }

    int endFrame() const;

    const int feature_id;
    int start_frame;
    std::vector<FeaturePerFrame> feature_per_frame;
    int used_num;
    bool is_outlier;
    bool is_margin;
    double estimated_depth;
    int solve_flag;         // 0 haven't solve yet; 1 solve succ; 2 solve fail;
    Eigen::Vector3d gt_p;
};

/**
 * 特征点管理类
 */
class FeatureManager {
public:
    FeatureManager(Eigen::Matrix3d _Rs[]);
    void setRic(Eigen::Matrix3d _ric[]);
    void clearState();
    int getFeatureCount();
    bool addFeatureCheckParallax(int frame_count, const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>& image, double td);
    void debugShow();
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);

    void setDepth(const Eigen::VectorXd& x);
    void removeFailures();
    void clearDepth(const Eigen::VectorXd& x);
    Eigen::VectorXd getDepthVector();
    void triangulate(Eigen::Vector3d Ps[], Eigen::Vector3d tic[], Eigen::Matrix3d ric[]);
    void removeBackShiftDepth(Eigen::Matrix3d& marg_R, Eigen::Vector3d& marg_P,
                              Eigen::Matrix3d& new_R, Eigen::Vector3d& new_P);
    void removeBack();
    void removeFront(int frame_count);
    void removeOutlier();
    std::list<FeaturePerId> feature;
    int last_track_num;

private:
    double compensatedParallax2(const FeaturePerId& it_per_id, int frame_count);
    const Eigen::Matrix3d *Rs;          // IMU的变换矩阵
    Eigen::Matrix3d ric[NUM_OF_CAM];    // 相机到参考系的变换矩阵
};

#endif //SIMPLE_VIO_FEATURE_MANAGER_H
