#include "feature_tracker.h"
#include <opencv2/opencv.hpp>
#include "parameters.h"
#include "utility/tic_toc.h"
#include "camodocal/camera_models/CameraFactory.h"

using namespace std;
using namespace camodocal;


int FeatureTracker::n_id = 0;

/// 判断特征点是否在图像内部
bool inBorder(const cv::Point2f& pt) {
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE
           && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

/// 去除无法跟踪的特征点
void reduceVector(vector<cv::Point2f>& v, vector<uchar>& status) {
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

/// 去除无法跟踪的特征点
void reduceVector(vector<int>& v, vector<uchar>& status) {
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

FeatureTracker::FeatureTracker() {
}

/**
 * 对跟踪点按照追踪次数进行排序
 * 使用mask进行类似非极大值抑制
 */
void FeatureTracker::setMask() {
    if(FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255)); // 默认进行此处操作

    /// 保存跟踪信息
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));

    /// 根据first进行排序
    sort(cnt_pts_id.begin(), cnt_pts_id.end(),
         [](const pair<int, pair<cv::Point2f, int>>& a,
            const pair<int, pair<cv::Point2f, int>>& b) {
        return a.first > b.first;
    });

    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    /// mask先全部设置为255（白色）填充，然后根据跟踪次数排序的特征点位置进行极大值抑制
    for (auto& it : cnt_pts_id) {
        if (mask.at<uchar>(it.second.first) == 255) {
            /// 按照跟踪次数多的特征点及对应信息优先加入,设置mask值为255
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            /// 将mask周围30的范围设置mask值为0,后续就不会被选择
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

/// 将新检测的特征点添加
void FeatureTracker::addPoints() {
    for (auto& p : n_pts) {
        forw_pts.push_back(p);
        ids.push_back(-1);      // 新提取的特征点id初始化为-1
        track_cnt.push_back(1); // 新提取的特征点跟踪次数初始化为1
    }
}

/**
 * 对图像使用光流法进行特征点跟踪
 * @param[in] _img      输入图像
 * @param[in] _cur_time 当前图像时间戳
 */
void FeatureTracker::readImage(const cv::Mat& _img, double _cur_time) {
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;

    if (EQUALIZE) { /// 太亮或者太暗,进行直方图均衡化处理,默认使用此选项
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        /// 均衡化处理
        clahe->apply(_img, img);
    } else
        img = _img;

    if (forw_img.empty()) { /// 如果当前帧图像为空,则说明是第一次读入,进行各个记录状态的初始化
        prev_img = cur_img = forw_img = img;
    } else { /// 只需更新当前帧数据
        forw_img = img;
    }

    forw_pts.clear();

    if (cur_pts.size() > 0) {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;
        // 使用LK金字塔光流跟踪对前一帧的cur_pts,得到跟踪的forw_pts
        // status标记了前一帧到forw_pts的跟踪状态,无法跟踪标记为0
        // 3+1层金字塔,21,21的搜索范围
        // 根据cur_pts在两幅图中找到forw_pts
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts,
                                 status, err, cv::Size(21, 21), 3);

        /// 将位于图像边界外的点标记为0
        for (int i = 0; i < int(forw_pts.size()); i++)
            if (status[i] && !inBorder(forw_pts[i]))
                status[i] = 0;
        /// 根据status,把跟踪失败的点剔除,所有记录中都要统一同步剔除
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);
    }

    /// 根据status,将跟踪成功的跟踪次数+1,次数大反映跟踪时间长
    for (auto& n : track_cnt)
        n++;

    if (PUB_THIS_FRAME) { // 默认是false
        /// 通过基础矩阵剔除outliers
        rejectWithF();
        TicToc t_m;
        /// 特征点周围进行30像素范围的极大值抑制
        setMask();
        TicToc t_t;
        /// 判断是否满足特征点需求,默认为150个
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0) {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;

            /** 在mask不为0的区域检测新的特征点
             *  参数说明:
             *      1.图像,2.存放角点的vector,3.能返回的角点的最大数量,
             *      4.角点判定质量水平阈值,5.角点之间最小距离,6.通过mask设置筛选区域
             *      7.计算协方差的窗口大小,8.是否使用harris角点算法,默认是使用shi-tomasi算法
             *      8.Harris角点的检测所需k值,默认为0.04
             */
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
        } else
            n_pts.clear();
        TicToc t_a;
        /// 将新检测到的特征点n_pts添加到记录中
        addPoints();
    }
    /// 更新pre,cur,forw信息
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    cur_img = forw_img;
    cur_pts = forw_pts;
    /// 使用不同的相机模型去畸变校正,转换为归一化坐标系,计算速度
    undistortedPoints();
    prev_time = cur_time;
}

/**
 * 通过基础矩阵剔除outliers
 */
void FeatureTracker::rejectWithF() {
    /// 确保能够构造基础矩阵
    if (forw_pts.size() >= 8) {
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++) {
            Eigen::Vector3d tmp_p;
            /// 根据不同的相机模型将二维坐标转换到三维坐标
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            /// 转化为归一化像素坐标
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        /// 求取基础矩阵 这里F_THRESHOLD为1个像素
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
    }
}

/// 更新特征点id
bool FeatureTracker::updateID(unsigned int i) {
    if (i < ids.size()) {
        if (ids[i] == -1)
            ids[i] = n_id++;
        return true;
    } else
        return false;
}

/// 读取相机内参
void FeatureTracker::readIntrinsicParameter(const string& calib_file) {
    cout << "reading paramerter of camera " << calib_file << endl;
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

/**
 * 显示去畸变校正后的特征点
 * @param name 图像帧名称
 */
void FeatureTracker::showUndistortion(const string& name) {
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++) {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
        }
    for (int i = 0; i < int(undistortedp.size()); i++) {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;

        if (pp.at<float>(1, 0) + 300 >= 0
            && pp.at<float>(1, 0) + 300 < ROW + 600
            && pp.at<float>(0, 0) + 300 >= 0
            && pp.at<float>(0, 0) + 300 < COL + 600) {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300)
                    = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        } else {
            // cout << "error" << endl;
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}

/**
 * 对角点图像坐标进行去畸变矫正,转换到归一化坐标系,计算每个角点的速度
 */
void FeatureTracker::undistortedPoints() {
    cur_un_pts.clear();
    cur_un_pts_map.clear();
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    for (unsigned int i = 0; i < cur_pts.size(); i++) {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
        // 根据相机模型将二维坐标转换到三维坐标
        m_camera->liftProjective(a, b);
        // 转换到归一化平面
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
    }
    // 计算特征点的运行速度,放入pts_velocity中
    if (!prev_un_pts_map.empty()) {
        double dt = cur_time - prev_time;
        pts_velocity.clear();
        for (unsigned int i = 0; i < cur_un_pts.size(); i++) {
            if (ids[i] != -1) {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);
                if (it != prev_un_pts_map.end()) {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                } else
                    pts_velocity.push_back(cv::Point2f(0, 0));
            } else {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    } else {
        for (unsigned int i = 0; i < cur_pts.size(); i++) {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    prev_un_pts_map = cur_un_pts_map;
}
