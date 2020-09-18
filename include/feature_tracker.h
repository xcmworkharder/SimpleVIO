#ifndef SIMPLE_VIO_FEATURE_TRACKER_H
#define SIMPLE_VIO_FEATURE_TRACKER_H

#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <map>
#include "camodocal/camera_models/Camera.h"

bool inBorder(const cv::Point2f& pt);
void reduceVector(std::vector<cv::Point2f>& v, std::vector<uchar>& status);
void reduceVector(std::vector<int>& v, std::vector<uchar>& status);

/**
 * 视觉前端:对每个相机进行角点LK光流跟踪
 */
class FeatureTracker {
public:
    FeatureTracker();

    void readImage(const cv::Mat& _img, double _cur_time);
    void setMask();
    void addPoints();
    bool updateID(unsigned int i);
    void readIntrinsicParameter(const std::string& calib_file);
    void showUndistortion(const std::string& name);
    void rejectWithF();
    void undistortedPoints();

    cv::Mat mask;                                           // 相机掩码
    cv::Mat fisheye_mask;                                   // 鱼眼相机mask,用来去除边缘噪声
    /// pre:上一次发布,cur:光流跟踪的前一帧,for:光流跟踪的后一帧
    cv::Mat prev_img, cur_img, forw_img;
    std::vector<cv::Point2f> n_pts;                         // 每一帧新提取的特征点
    std::vector<cv::Point2f> prev_pts, cur_pts, forw_pts;
    std::vector<cv::Point2f> prev_un_pts, cur_un_pts;
    std::vector<cv::Point2f> pts_velocity;
    std::vector<int> ids;
    std::vector<int> track_cnt;
    std::map<int, cv::Point2f> cur_un_pts_map;
    std::map<int, cv::Point2f> prev_un_pts_map;
    camodocal::CameraPtr m_camera;                          // 相机模型
    double cur_time;
    double prev_time;

    static int n_id;
};

#endif //SIMPLE_VIO_FEATURE_TRACKER_H
