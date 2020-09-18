#ifndef SIMPLE_VIO_INITIAL_SFM_H
#define SIMPLE_VIO_INITIAL_SFM_H

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Dense>

/// 每个路标点由多个连续图像观测到
struct SFMFeature {
    bool state; // 特征点的状态（是否被三角化）
    int id;     // 特征点id
    // 所有观测到该特征点的图像帧ID和在图像上对应的坐标
    std::vector<std::pair<int, Eigen::Vector2d>> observation;
    double position[3]; // landmark的3d坐标,这样定义便于ceres优化使用
    double depth;       // 深度
};

/// 用于ceres优化的代价类, 代价仿函数和ceres代价函数生成函数
struct ReprojectionError3D {
    ReprojectionError3D(double _observed_u, double _observed_v)
            : observed_u(_observed_u), observed_v(_observed_v) { }

    template <typename T>
    bool operator()(const T* const camera_R, const T* const camera_T,
                    const T* point, T* residuals) const {
        T p[3];
        ceres::QuaternionRotatePoint(camera_R, point, p); // 将point通过camera_R旋转变换为p
        p[0] += camera_T[0];
        p[1] += camera_T[1];
        p[2] += camera_T[2];
        T xp = p[0] / p[2];
        T yp = p[1] / p[2];
        residuals[0] = xp - T(observed_u);
        residuals[1] = yp - T(observed_v);
        return true;
    }

    static ceres::CostFunction* Create(const double observed_x,
                                       const double observed_y) {
        return (new ceres::AutoDiffCostFunction<
                ReprojectionError3D, 2, 4, 3, 3>(
                new ReprojectionError3D(observed_x,observed_y)));
    }

    double observed_u;
    double observed_v;
};

// 滑窗内所有帧的位姿计算
class GlobalSFM {
public:
    GlobalSFM();
    bool construct(int frame_num, Eigen::Quaterniond* q, Eigen::Vector3d* T, int l,
                   const Eigen::Matrix3d& relative_R, const Eigen::Vector3d& relative_T,
                   std::vector<SFMFeature>& sfm_f,
                   std::map<int, Eigen::Vector3d>& sfm_tracked_points);

private:
    bool solveFrameByPnP(Eigen::Matrix3d& R_initial, Eigen::Vector3d& P_initial,
                         int i, std::vector<SFMFeature>& sfm_f);
    void triangulatePoint(Eigen::Matrix<double, 3, 4>& Pose0,
                          Eigen::Matrix<double, 3, 4>& Pose1,
                          Eigen::Vector2d& point0, Eigen::Vector2d& point1,
                          Eigen::Vector3d& point_3d);
    void triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4>& Pose0,
                              int frame1, Eigen::Matrix<double, 3, 4>& Pose1,
                              std::vector<SFMFeature> &sfm_f);
    int feature_num; // 特征点的数量
};

#endif //SIMPLE_VIO_INITIAL_SFM_H
