#include "initialization/initial_sfm.h"
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <map>

using namespace Eigen;
using namespace std;

GlobalSFM::GlobalSFM() {

}

/**
 * 三角化两帧间某个对应特征点坐标(包括深度)
 * @param[in] Pose0 第一帧world2camera
 * @param[in] Pose1 第二帧world2camera
 * @param[in] point0 第一帧中归一化坐标
 * @param[in] point1 第二帧中归一化坐标
 * @param[out] point_3d 三角化得出的landmark结果
 */
void GlobalSFM::triangulatePoint(Matrix<double, 3, 4>& Pose0,
                                 Matrix<double, 3, 4>& Pose1,
                                 Vector2d& point0, Vector2d& point1,
                                 Vector3d& point_3d) {
    Matrix4d design_matrix = Matrix4d::Zero();
    design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
    design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
    design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
    design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
    Vector4d triangulated_point;
    triangulated_point =
            design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    point_3d(0) = triangulated_point(0) / triangulated_point(3);
    point_3d(1) = triangulated_point(1) / triangulated_point(3);
    point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

/**
 * 通过找到一些特征点在某帧图像的投影,使用pnp得出特征点对应的参考系到相机系的变换R和t
 * @param[out] R_initial
 * @param[out] P_initial
 * @param[in] i
 * @param[in] sfm_f
 * @return
 */
bool GlobalSFM::solveFrameByPnP(Matrix3d& R_initial, Vector3d& P_initial, int i,
                                vector<SFMFeature>& sfm_f) {
    vector<cv::Point2f> pts_2_vector;
    vector<cv::Point3f> pts_3_vector;
    // 遍历所有特征点, 找到第i帧图像上某个特征点对应的像素投影
    for (int j = 0; j < feature_num; ++j) {
        // 如果未被三角化计算过,跳过
        if (sfm_f[j].state != true) continue;
        Vector2d point2d;
        // 在所有第j个特征的观测图像帧中找到帧id为i的图像
        for (int k = 0; k < (int)sfm_f[j].observation.size(); ++k) {
            if (sfm_f[j].observation[k].first == i) {
                // 获取特征点对应的像素投影点
                Vector2d img_pts = sfm_f[j].observation[k].second;
                cv::Point2f pts_2(img_pts(0), img_pts(1));
                pts_2_vector.push_back(pts_2);
                cv::Point3f pts_3(sfm_f[j].position[0], sfm_f[j].position[1],
                                  sfm_f[j].position[2]);
                pts_3_vector.push_back(pts_3);
                break;
            }
        }
    }
    if (int(pts_2_vector.size()) < 15) {
        cout << "unstable features tracking, please slowly move you device!" << endl;
        if (int(pts_2_vector.size()) < 10) return false;
    }
    cv::Mat r, rvec, t, D, tmp_r;
    cv::eigen2cv(R_initial, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_initial, t);
    // 相机内参K使用单位矩阵
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    bool pnp_succ;
    // from model system to cameral system, 畸变参数取为0, 采用efficient pnp方法
    pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);
    if (!pnp_succ) {
        return false;
    }
    // 用罗德里格斯公式将旋转矢量变为旋转矩阵
    cv::Rodrigues(rvec, r);
    MatrixXd R_pnp;
    cv::cv2eigen(r, R_pnp);
    MatrixXd T_pnp;
    cv::cv2eigen(t, T_pnp);
    R_initial = R_pnp;
    P_initial = T_pnp;
    return true;
}

/**
 * 三角化处理两帧的所有匹配点
 * @param[in] frame0
 * @param[in] Pose0
 * @param[in] frame1
 * @param[in] Pose1
 * @param[out] sfm_f
 */
void GlobalSFM::triangulateTwoFrames(int frame0, Matrix<double, 3, 4>& Pose0,
                                     int frame1, Matrix<double, 3, 4>& Pose1,
                                     vector<SFMFeature>& sfm_f) {
    // assert(frame0 != frame1);
    if (frame0 == frame1) {
        return;
    }
    for (int j = 0; j < feature_num; ++j) {
        // 如果这个landmark已经三角化过,则跳过
        if (sfm_f[j].state == true) continue;
        bool has_0 = false, has_1 = false;
        Vector2d point0;
        Vector2d point1;
        // 查找在frame0和frame1中同时观测到landmark_j的投影坐标
        for (int k = 0; k < (int)sfm_f[j].observation.size(); ++k) {
            if (sfm_f[j].observation[k].first == frame0) {
                point0 = sfm_f[j].observation[k].second;
                has_0 = true;
            }
            if (sfm_f[j].observation[k].first == frame1) {
                point1 = sfm_f[j].observation[k].second;
                has_1 = true;
            }
        }
        // 根据两个投影坐标三角化出landmark的三维坐标
        if (has_0 && has_1) {
            Vector3d point_3d;
            triangulatePoint(Pose0, Pose1, point0, point1, point_3d);
            sfm_f[j].state = true;
            sfm_f[j].position[0] = point_3d(0);
            sfm_f[j].position[1] = point_3d(1);
            sfm_f[j].position[2] = point_3d(2);
        }
    }
}

/**
 * @brief   纯视觉sfm，求解窗口中的所有图像帧的位姿和特征点坐标
 * @param[in]   frame_num	窗口总帧数（frame_count + 1）
 * @param[out]  q 	窗口内图像帧的旋转四元数q（相对于第l帧）
 * @param[out]	T 	窗口内图像帧的平移向量T（相对于第l帧）
 * @param[in]  	l 	第l帧,作为参考帧,选择特征点较多的关键帧
 * @param[in]  	relative_R	当前帧到第l帧的旋转矩阵
 * @param[in]  	relative_T 	当前帧到第l帧的平移向量
 * @param[in]  	sfm_f		所有特征点, 包含投影坐标,未计算3d坐标
 * @param[out]  sfm_tracked_points 所有在sfm中三角化的特征点ID和坐标
 * @return  bool true:sfm求解成功
*/
bool GlobalSFM::construct(int frame_num, Quaterniond* q, Vector3d* T, int l,
                          const Matrix3d& relative_R, const Vector3d& relative_T,
                          vector<SFMFeature>& sfm_f,
                          map<int, Vector3d>& sfm_tracked_points) {
    feature_num = sfm_f.size();
    // 以第l帧到参考系的变换
    q[l].w() = 1;
    q[l].x() = 0;
    q[l].y() = 0;
    q[l].z() = 0;
    T[l].setZero();
    // 索引fame_num - 1为当前帧
    q[frame_num - 1] = q[l] * Quaterniond(relative_R);
    T[frame_num - 1] = relative_T;

    // 定义滑窗内各帧的旋转变换阵数组
    Matrix3d c_Rotation[frame_num];
    Vector3d c_Translation[frame_num];
    Quaterniond c_Quat[frame_num];
    double c_rotation[frame_num][4];    // 四元数 用于ceres优化使用
    double c_translation[frame_num][3]; // 平移量
    // 表示第l帧到每一帧的变换矩阵
    Matrix<double, 3, 4> Pose[frame_num];

    // 初始化参考系到第l帧的信息
    c_Quat[l] = q[l].inverse();
    c_Rotation[l] = c_Quat[l].toRotationMatrix();
    c_Translation[l] = -1 * (c_Rotation[l] * T[l]);
    Pose[l].block<3, 3>(0, 0) = c_Rotation[l];
    Pose[l].block<3, 1>(0, 3) = c_Translation[l];

    // 初始化参考系到当前(frame_num - 1)帧的变换信息
    c_Quat[frame_num - 1] = q[frame_num - 1].inverse();
    c_Rotation[frame_num - 1] = c_Quat[frame_num - 1].toRotationMatrix();
    c_Translation[frame_num - 1] = -1 * (c_Rotation[frame_num - 1] * T[frame_num - 1]);
    Pose[frame_num - 1].block<3, 3>(0, 0) = c_Rotation[frame_num - 1];
    Pose[frame_num - 1].block<3, 1>(0, 3) = c_Translation[frame_num - 1];

    // 1、先三角化第l帧（参考帧）与第frame_num-1帧（当前帧）的路标点
    // 2、pnp求解从第l+1开始的每一帧到第l帧的变换矩阵R_initial, P_initial，保存在Pose中
    // 并与当前帧进行三角化
    for (int i = l; i < frame_num - 1 ; i++) {
        if (i > l) {
            Matrix3d R_initial = c_Rotation[i - 1];
            Vector3d P_initial = c_Translation[i - 1];
            if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
                return false;
            // l帧(参考帧)到第i帧的变换
            c_Rotation[i] = R_initial;
            c_Translation[i] = P_initial;
            c_Quat[i] = c_Rotation[i];
            Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
            Pose[i].block<3, 1>(0, 3) = c_Translation[i];
        }
        // 固定frame_num -1即当前帧, 对第i帧进行三角化处理
        triangulateTwoFrames(i, Pose[i], frame_num - 1, Pose[frame_num - 1], sfm_f);
    }

    // 3、固定第l帧,第l+1到frame_num -2中未三角化的帧再进行三角化
    for (int i = l + 1; i < frame_num - 1; i++) {
        triangulateTwoFrames(l, Pose[l], i, Pose[i], sfm_f);
    }

    // 4、固定第l帧,PNP求解从第l-1到第0帧的每一帧与第l帧之间的变换矩阵，并进行三角化
    for (int i = l - 1; i >= 0; i--) {
        //solve pnp
        Matrix3d R_initial = c_Rotation[i + 1];
        Vector3d P_initial = c_Translation[i + 1];
        if (!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
            return false;
        c_Rotation[i] = R_initial;
        c_Translation[i] = P_initial;
        c_Quat[i] = c_Rotation[i];
        Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
        Pose[i].block<3, 1>(0, 3) = c_Translation[i];
        //triangulate
        triangulateTwoFrames(i, Pose[i], l, Pose[l], sfm_f);
    }

    // 5、三角化其他未恢复的特征点
    // 至此得到了滑动窗口中所有图像帧的位姿以及特征点的3d坐标
    for (int j = 0; j < feature_num; j++) {
        if (sfm_f[j].state == true) continue;
        if ((int)sfm_f[j].observation.size() >= 2) {
            Vector2d point0, point1;
            int frame_0 = sfm_f[j].observation[0].first;
            point0 = sfm_f[j].observation[0].second;
            int frame_1 = sfm_f[j].observation.back().first;
            point1 = sfm_f[j].observation.back().second;
            Vector3d point_3d;
            triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1, point_3d);
            sfm_f[j].state = true;
            sfm_f[j].position[0] = point_3d(0);
            sfm_f[j].position[1] = point_3d(1);
            sfm_f[j].position[2] = point_3d(2);
        }
    }

    // 6、将以上得出的位姿和landmark点结果使用ceres进行全局BA优化
    ceres::Problem problem;
    ceres::LocalParameterization* local_parameterization =
            new ceres::QuaternionParameterization();
    for (int i = 0; i < frame_num; i++) {
        // double array for ceres
        c_translation[i][0] = c_Translation[i].x();
        c_translation[i][1] = c_Translation[i].y();
        c_translation[i][2] = c_Translation[i].z();
        c_rotation[i][0] = c_Quat[i].w();
        c_rotation[i][1] = c_Quat[i].x();
        c_rotation[i][2] = c_Quat[i].y();
        c_rotation[i][3] = c_Quat[i].z();
        // local_parameterization能够解决过参数问题,将自由度4优化为本地维度3
        problem.AddParameterBlock(c_rotation[i], 4, local_parameterization);
        problem.AddParameterBlock(c_translation[i], 3);
        if (i == l) {
            // 优化过程中不变
            problem.SetParameterBlockConstant(c_rotation[i]);
        }
        if (i == l || i == frame_num - 1) {
            problem.SetParameterBlockConstant(c_translation[i]);
        }
    }

    for (int i = 0; i < feature_num; i++) {
        // 如果特征点没有被2幅以上的图像同时观测到,则不会被三角化,不能使用
        if (sfm_f[i].state != true) continue;
        for (int j = 0; j < int(sfm_f[i].observation.size()); j++) {
            int l = sfm_f[i].observation[j].first;
            ceres::CostFunction* cost_function = ReprojectionError3D::Create(
                    sfm_f[i].observation[j].second.x(),
                    sfm_f[i].observation[j].second.y());
            problem.AddResidualBlock(cost_function, NULL, c_rotation[l],
                                     c_translation[l], sfm_f[i].position);
        }

    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.minimizer_progress_to_stdout = true;
    options.max_solver_time_in_seconds = 0.2;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //std::cout << summary.BriefReport() << "\n";
    if (summary.termination_type == ceres::CONVERGENCE
        || summary.final_cost < 5e-03) {
        //cout << "vision only BA converge" << endl;
    } else {
        //cout << "vision only BA not converge " << endl;
        return false;
    }

    // 这里得到的是第l帧坐标系到各帧的变换矩阵，应将其转变为各帧在第l帧坐标系上的位姿
    for (int i = 0; i < frame_num; i++) {
        q[i].w() = c_rotation[i][0];
        q[i].x() = c_rotation[i][1];
        q[i].y() = c_rotation[i][2];
        q[i].z() = c_rotation[i][3];
        q[i] = q[i].inverse();
    }
    // 可与上边代码合并到一个for循环
    for (int i = 0; i < frame_num; i++) {
        T[i] = -1 * (q[i] * Vector3d(c_translation[i][0], c_translation[i][1], c_translation[i][2]));
    }
    for (int i = 0; i < (int)sfm_f.size(); i++) {
        if(sfm_f[i].state)
            sfm_tracked_points[sfm_f[i].id] = Vector3d(sfm_f[i].position[0],
                                                       sfm_f[i].position[1],
                                                       sfm_f[i].position[2]);
    }
    return true;
}
