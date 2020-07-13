#include "feature_manager.h"

using namespace std;

// 获取某特征点关联的最后一个图像帧的索引
int FeaturePerId::endFrame() const {
    return start_frame + feature_per_frame.size() - 1;
}

/**
 * 特征管理类的构造函数
 * @param _Rs 初始化
 */
FeatureManager::FeatureManager(Eigen::Matrix3d _Rs[]) : Rs(_Rs) {
    for (int i = 0; i < NUM_OF_CAM; i++)
        ric[i].setIdentity();
}

// 设置相机到IMU的旋转矩阵
void FeatureManager::setRic(Eigen::Matrix3d _ric[]) {
    for (int i = 0; i < NUM_OF_CAM; i++) {
        ric[i] = _ric[i];
    }
}

// 清除(特征点-图像帧)链表
void FeatureManager::clearState() {
    feature.clear();
}

// 获取窗口中被跟踪的特征数量
int FeatureManager::getFeatureCount() {
    int cnt = 0;
    for (auto& it : feature) {
        it.used_num = it.feature_per_frame.size();
        // 每个特征对应的帧数大于2,且对应的开始帧索引小于窗口尺寸-2,则表示特征被跟踪
        if (it.used_num >= 2 && it.start_frame < WINDOW_SIZE - 2) {
            cnt++;
        }
    }
    return cnt;
}

/**
 * 描述:把特征点放入list中
 *     计算每一个特征点的跟踪次数和其与次新帧,次次新帧的视差
 *     判断次新帧是否为关键帧
 * @param frame_count 窗口内帧的个数
 * @param image {feature_id, [{camera_id, [x, y, z, u, v, vx, vy]}, ...]}
 * @param td IMU和camera的同步时间差
 * @return true:次新帧为关键帧, false:非关键帧
 */
bool FeatureManager::addFeatureCheckParallax(int frame_count,
                                             const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>& image,
                                             double td) {
    double parallax_sum = 0;
    int parallax_num = 0;
    last_track_num = 0;
    // 把map的所有特征点放入list中
    for (auto& id_pts : image) {
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td); // 生成第一个相机位置信息

        // 寻找list中是否存在feature_id
        int feature_id = id_pts.first;
        auto it = find_if(feature.begin(), feature.end(),
                          [feature_id](const FeaturePerId& it) {
                            return it.feature_id == feature_id;
                          }
        );

        // 如果不存在,则创建一个图像帧
        if (it == feature.end()) {
            feature.push_back(FeaturePerId(feature_id, frame_count));
            feature.back().feature_per_frame.push_back(f_per_fra);
        } else if (it->feature_id == feature_id) {
            it->feature_per_frame.push_back(f_per_fra);
            last_track_num++;
        }
    }

    if (frame_count < 2 || last_track_num < 20) // 这种情况下将次新帧设为关键帧
        return true;

    // 计算每个特征点在次新帧和次次新帧的视差
    for (auto& it_per_id : feature) {
        if (it_per_id.start_frame <= frame_count - 2 &&
        it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1)
        {
            parallax_sum += compensatedParallax2(it_per_id, frame_count);
            parallax_num++;
        }
    }

    if (parallax_num == 0) { // 如果新帧中没有历史特征,则设为关键帧
        return true;
    } else { // 否则,如果视差足够大,也设置为关键帧
        return parallax_sum / parallax_num >= MIN_PARALLAX;
    }
}

// 得到frame_count_l与frame_count_r两帧上共有的特征点在这两帧上的位置投影对
vector<pair<Eigen::Vector3d, Eigen::Vector3d>>
FeatureManager::getCorresponding(int frame_count_l, int frame_count_r) {
    vector<pair<Eigen::Vector3d, Eigen::Vector3d>> corres;
    for (auto& it : feature) {
        // 如果某个feature出现的帧索引范围包含查找范围,进行记录查找范围边界点对
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r) {
            Eigen::Vector3d a = Eigen::Vector3d::Zero(),
                            b = Eigen::Vector3d::Zero();
            // 获取区间左右边界对应的特征缓存图像的索引
            int idx_l = frame_count_l - it.start_frame;
            int idx_r = frame_count_r - it.start_frame;
            a = it.feature_per_frame[idx_l].point;
            b = it.feature_per_frame[idx_r].point;
            corres.push_back(make_pair(a, b));
        }
    }
    return corres;
}

// 使用x设置特征点的逆深度估计值
void FeatureManager::setDepth(const Eigen::VectorXd& x) {
    int feature_index = -1;
    for (auto& it_per_id : feature) {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        // 不满足最少2个观测,且开始帧小于窗口尺寸-2,则无法计算
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        it_per_id.estimated_depth = 1.0 / x(++feature_index);
        if (it_per_id.estimated_depth < 0) {
            it_per_id.solve_flag = 2; // 失败估计
        }
        else
            it_per_id.solve_flag = 1; // 成功估计
    }
}

// 剔除feature估计失败的点
void FeatureManager::removeFailures() {
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next) {
        it_next++;
        if (it->solve_flag == 2)
            feature.erase(it);
    }
}

// 使用信息:x清除深度
void FeatureManager::clearDepth(const Eigen::VectorXd& x) {
    int feature_index = -1;
    for (auto& it_per_id : feature) {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth = 1.0 / x(++feature_index);
    }
}

// 获取深度列表
Eigen::VectorXd FeatureManager::getDepthVector() {
    Eigen::VectorXd dep_vec(getFeatureCount());
    int feature_index = -1;
    for (auto& it_per_id : feature) {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
#if 1
        dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
#else
        dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
    }
    return dep_vec;
}

// 对特征点进行三角化求深度(SVD分解)
void FeatureManager::triangulate(Eigen::Vector3d Ps[],
                                 Eigen::Vector3d tic[],
                                 Eigen::Matrix3d ric[]) {
    for (auto& it_per_id : feature) {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        if (it_per_id.estimated_depth > 0) // 如果已经大于0,则不用估计
            continue;
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        //assert(NUM_OF_CAM == 1);
        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
        int svd_idx = 0;

        Eigen::Matrix<double, 3, 4> P0;
        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];    // 相机到参考系
        Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];                // 相机到参考系
        P0.leftCols<3>() = Eigen::Matrix3d::Identity();
        P0.rightCols<1>() = Eigen::Vector3d::Zero();

        for (auto& it_per_frame : it_per_id.feature_per_frame) {
            imu_j++;
            Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
            Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];
            Eigen::Vector3d t = R0.transpose() * (t1 - t0);
            Eigen::Matrix3d R = R0.transpose() * R1;
            Eigen::Matrix<double, 3, 4> P;
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;
            Eigen::Vector3d f = it_per_frame.point.normalized();
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

            if (imu_i == imu_j)
                continue;
        }
        //assert(svd_idx == svd_A.rows());
        Eigen::Vector4d svd_V =
                Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        double svd_method = svd_V[2] / svd_V[3];

        it_per_id.estimated_depth = svd_method;

        if (it_per_id.estimated_depth < 0.1) {
            it_per_id.estimated_depth = INIT_DEPTH;
        }
    }
}

// 剔除外点
void FeatureManager::removeOutlier() {
    int i = -1;
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next) {
        it_next++;
        i += it->used_num != 0;
        if (it->used_num != 0 && it->is_outlier == true) {
            feature.erase(it);
        }
    }
}

/**
 * 边缘化最老帧
 * @param marg_R 被边缘化的姿态
 * @param marg_P 被边缘化的位置
 * @param new_R  新一帧的姿态
 * @param new_P  新一帧的位置
 */
void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d& marg_R,
                                          Eigen::Vector3d& marg_P,
                                          Eigen::Matrix3d& new_R,
                                          Eigen::Vector3d& new_P) {
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next) {
        it_next++;
        if (it->start_frame != 0) { // 特征点起始帧不是最老帧,则将帧号减一
            it->start_frame--;
        } else {
            // 特征点对应的起始帧是最老帧,则将剔除特征点对应的起始帧
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point;
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            // 如果特征点只在最老帧被观测,则直接剔除该特征点
            if (it->feature_per_frame.size() < 2) {
                feature.erase(it);
                continue;
            } else {
                // pts_i:特征点在最老帧坐标系下的三维坐标
                Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
                // w_pts_i:特征点在世界坐标系的三维坐标
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                // 转化到下一帧坐标系
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
                double dep_j = pts_j(2);
                // 更新估计深度
                if (dep_j > 0) {
                    it->estimated_depth = dep_j;
                } else {
                    it->estimated_depth = INIT_DEPTH;
                }
            }
        }
    }
}

// 边缘化最老帧,直接将特征点保存的帧号向前移动
void FeatureManager::removeBack() {
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next) {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else {
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

// 边缘化次新帧,对特征点在次新帧的信息进行移除处理
void FeatureManager::removeFront(int frame_count) {
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next) {
        it_next++;
        // 起始帧为最新帧的滑动成次新帧
        if (it->start_frame == frame_count) {
            it->start_frame--;
        } else {
            int j = WINDOW_SIZE - 1 - it->start_frame;
            // 如果在次新帧前跟踪结束,则跳过
            if (it->endFrame() < frame_count - 1)
                continue;
            // 否则说明次新帧仍被跟踪,则剔除次新帧对
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
            // 如果feature_frame为空,则直接剔除特征点
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

// 计算某个特征点在次新帧和次次新帧的视差
double FeatureManager::compensatedParallax2(const FeaturePerId& it_per_id,
                                            int frame_count) {
    // 次次新帧
    const FeaturePerFrame& frame_i =
            it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
    // 次新帧
    const FeaturePerFrame& frame_j =
            it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

    double ans = 0;
    Eigen::Vector3d p_j = frame_j.point;

    double u_j = p_j(0);
    double v_j = p_j(1);

    Eigen::Vector3d p_i = frame_i.point;
    Eigen::Vector3d p_i_comp;

    p_i_comp = p_i;
    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j;

    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));
    return ans;
}