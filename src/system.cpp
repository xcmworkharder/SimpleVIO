#include "system.h"
#include "utility/tic_toc.h"

using namespace std;
using namespace cv;


// 为系统进行初始化设置
System::System(const string& sConfig_file_) : bStart_backend(true) {
    string sConfig_file = sConfig_file_ + "euroc_config.yaml";
    cout << "1: System() sConfig_file: " << sConfig_file << endl;
    // 读取各种配置参数
    readParameters(sConfig_file);
    // 读取相机参数,创建相机模型,这里使用针孔相机
    trackerData[0].readIntrinsicParameter(sConfig_file);
    // 为估计器设置参数
    estimator.setParameter();
    // 设置数据输出路径
//    ofs_pose.open("./pose_output.txt", fstream::app | fstream::out);
    ofs_pose.open("./pose_output.txt", fstream::out);
    if(!ofs_pose.is_open()) {
        cerr << "ofs_pose is not open" << endl;
    }
    cout << "2: System() end" << endl;
}

// 进行系统析构处理
System::~System() {
    // 停止线程
    bStart_backend = false;
    pangolin::Quit();

    // 清空缓冲区
    m_buf.lock();
    while (!feature_buf.empty())
        feature_buf.pop();
    while (!imu_buf.empty())
        imu_buf.pop();
    m_buf.unlock();

    m_estimator.lock();
    estimator.clearState();
    m_estimator.unlock();
    // 关闭输出文件
    ofs_pose.close();
}

// 发布图像信息
void System::pubImageData(double dStampSec, Mat& img) {
    // 如果是图像第一帧,则标记为初始特征,不使用该帧(因为没有光流速度)
    if (!init_feature) {
        cout << "1 PubImageData skip the first detected feature, "
                "which doesn't contain optical flow speed" << endl;
        init_feature = 1;
        return;
    }

    // 如果是第二帧,则记录时间
    if (first_image_flag) {
        cout << "2 PubImageData first_image_flag" << endl;
        first_image_flag = false;
        first_image_time = dStampSec;
        last_image_time = dStampSec;
        return;
    }

    // 剔除不稳定的跟踪帧(当前帧与上一帧时间超过一秒, 或者比上一帧时间还早)
    if (dStampSec - last_image_time > 1.0 || dStampSec < last_image_time) {
        cerr << "3 PubImageData image discontinue! reset the feature tracker!" << endl;
        // 重置第一帧位置
        first_image_flag = true;
        // 设置上一帧时间戳为0
        last_image_time = 0;
        pub_count = 1;
        return;
    }

    // 正常记录,用当前时间更新上一帧记录
    last_image_time = dStampSec;

    // 控制频率与输出帧数保持在PREQ
    if (round(1.0 * pub_count / (dStampSec - first_image_time)) <= FREQ) {
        PUB_THIS_FRAME = true;
        // 如果频率小于但已经接近FREQ,则发布当前帧,并重置发布频率控制参数
        if (abs(1.0 * pub_count / (dStampSec - first_image_time) - FREQ)
            < 0.01 * FREQ) {
            first_image_time = dStampSec;
            pub_count = 0;
        }
    } else { // 如果输出的太多,超过固定频率,则当前帧不发布
        PUB_THIS_FRAME = false;
    }

    TicToc t_r;
    // cout << "3 PubImageData t : " << dStampSec << endl;
    // 使用光流进行特征点跟踪
    trackerData[0].readImage(img, dStampSec);

    // 对新跟踪到的特征点在原有特征点id基础上进行id更新
    for (unsigned int i = 0;; i++) {
        bool completed = false;
        completed |= trackerData[0].updateID(i);
        // 如果超过特征点个数,则没有可更新的,跳出循环
        if (!completed)
            break;
    }

    // 发布当前帧
    if (PUB_THIS_FRAME) {
        // 帧数+1
        pub_count++;
        shared_ptr<IMG_MSG> feature_points(new IMG_MSG());
        feature_points->header = dStampSec;
        vector<set<int>> hash_ids(NUM_OF_CAM);
        // 为不同相机进行当前帧图像的特征点数据进行赋值
        for (int i = 0; i < NUM_OF_CAM; i++) {
            auto& un_pts = trackerData[i].cur_un_pts;
            auto& cur_pts = trackerData[i].cur_pts;
            auto& ids = trackerData[i].ids;
            auto& pts_velocity = trackerData[i].pts_velocity;
            for (unsigned int j = 0; j < ids.size(); j++) {
                if (trackerData[i].track_cnt[j] > 1) {  // 当跟踪次数大于1时
                    int p_id = ids[j];
                    hash_ids[i].insert(p_id);
                    double x = un_pts[j].x;
                    double y = un_pts[j].y;
                    double z = 1;
                    feature_points->points.push_back(Vector3d(x, y, z));
                    feature_points->id_of_point.push_back(p_id * NUM_OF_CAM + i);
                    feature_points->u_of_point.push_back(cur_pts[j].x);
                    feature_points->v_of_point.push_back(cur_pts[j].y);
                    feature_points->velocity_x_of_point.push_back(pts_velocity[j].x);
                    feature_points->velocity_y_of_point.push_back(pts_velocity[j].y);
                }
            }
            // 跳过第一帧图像
            if (!init_pub) {
                cout << "4 PubImage init_pub skip the first image!" << endl;
                init_pub = 1;
            } else {
                // 将图像的特征点信息发布出去
                m_buf.lock();
                feature_buf.push(feature_points);
                m_buf.unlock();
                // 通过其他阻塞线程进行处理
                con.notify_one();
            }
        }
    }

    // 显示跟踪的特征点信息
#ifdef __linux__
    cv::Mat show_img;
    cv::cvtColor(img, show_img, CV_GRAY2RGB); // 将img显示在show_img上
    if (SHOW_TRACK) {
        for (unsigned int j = 0; j < trackerData[0].cur_pts.size(); j++) {
            double len = min(1.0, 1.0 * trackerData[0].track_cnt[j] / WINDOW_SIZE);
            // 根据跟踪次数显示特征点, 跟踪次数多的显示为红色,次数少的蓝色 color:BGR
            cv::circle(show_img, trackerData[0].cur_pts[j], 2,
                       cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
        }

        cv::namedWindow("IMAGE", CV_WINDOW_AUTOSIZE);
        cv::imshow("IMAGE", show_img);
        cv::waitKey(1);
    }
#endif
}

// 获取IMU和图像量测信息, [(多个IMU,1个图像),(多个IMU,1个图像),....]
vector<pair<vector<ImuConstPtr>, ImgConstPtr>> System::getMeasurements() {
    vector<pair<vector<ImuConstPtr>, ImgConstPtr>> measurements;
    while (true) {
        // 如果IMU和图像特征信息其中之一为空,则直接返回
        if (imu_buf.empty() || feature_buf.empty()) {
            // cerr << "1 imu_buf.empty() || feature_buf.empty()" << endl;
            return measurements;
        }
        // 如果不满足imu的结束时间戳大于图像开始+估计时间则等待数量增加,直接返回
        if (!(imu_buf.back()->header > feature_buf.front()->header + estimator.td)) {
            cerr << "wait for imu, only should happen at the beginning sum_of_wait: "
                 << sum_of_wait << endl;
            sum_of_wait++;
            return measurements;
        }
        // 如果不满足imu开始时间戳小于图像开始+估计时间,则feature_buf弹出front(),继续提取
        if (!(imu_buf.front()->header < feature_buf.front()->header + estimator.td)) {
            cerr << "throw img, only should happen at the beginning" << endl;
            feature_buf.pop();
            continue;
        }
        // 提取图像的第一帧
        ImgConstPtr img_msg = feature_buf.front();
        feature_buf.pop();
        // 根据图像第一帧的时间戳,提取imu信息
        vector<ImuConstPtr> IMUs;
        while (imu_buf.front()->header < img_msg->header + estimator.td) {
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }
        // 提取的最后一个imu信息时间戳要超过图像的时间戳
        IMUs.emplace_back(imu_buf.front());
        if (IMUs.empty()) {
            cerr << "no imu between two image" << endl;
        }
        measurements.emplace_back(IMUs, img_msg); // 增加IMU和img_msg的量测信息
    }
    return measurements;
}

// 发布Imu数据
void System::pubImuData(double dStampSec, const Eigen::Vector3d& vGyr,
                        const Eigen::Vector3d& vAcc) {
    shared_ptr<IMU_MSG> imu_msg(new IMU_MSG());
    imu_msg->header = dStampSec;
    imu_msg->linear_acceleration = vAcc;
    imu_msg->angular_velocity = vGyr;

    if (dStampSec <= last_imu_t) { // 新的时间戳要大于上一个imu时间戳,否则异常退出
        cerr << "imu message in disorder!" << endl;
        return;
    }
    last_imu_t = dStampSec;
    m_buf.lock();
    imu_buf.push(imu_msg);
    m_buf.unlock();
    // 发布imu数据,通知其他线程
    con.notify_one();
}

// 开始进行后端处理
void System::processBackEnd() {
    cout << "1 ProcessBackEnd start" << endl;
    // 如果设置后端启动,则进行循环处理
    while (bStart_backend) {
        // cout << "1 process()" << endl;
        vector<pair<vector<ImuConstPtr>, ImgConstPtr>> measurements;

        // 先获取imu和图像量测信息
        unique_lock<mutex> lk(m_buf);
        con.wait(lk, [&] {
            return (measurements = getMeasurements()).size() != 0;
        });
        if (measurements.size() > 1) {
            cout << "1 getMeasurements size: " << measurements.size()
                 << " imu sizes: " << measurements[0].first.size()
                 << " feature_buf size: " <<  feature_buf.size()
                 << " imu_buf size: " << imu_buf.size() << endl;
        }
        lk.unlock();

        // 开始进行估计处理
        m_estimator.lock();
        for (auto& measurement : measurements) {
            auto img_msg = measurement.second; // 获取图像信息
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            for (auto& imu_msg : measurement.first) {
                double t = imu_msg->header;
                double img_t = img_msg->header + estimator.td;

                if (t <= img_t) { // 先对时间戳小于图像时间戳部分IMU进行预积分处理
                    // 默认为-1
                    if (current_time < 0)
                        current_time = t;
                    double dt = t - current_time;
                    assert(dt >= 0);
                    current_time = t;
                    dx = imu_msg->linear_acceleration.x();
                    dy = imu_msg->linear_acceleration.y();
                    dz = imu_msg->linear_acceleration.z();
                    rx = imu_msg->angular_velocity.x();
                    ry = imu_msg->angular_velocity.y();
                    rz = imu_msg->angular_velocity.z();
                    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                } else { // 最后一个IMU时间超过图像帧时间,先进行IMU数据插值处理,再进行预积分
                    double dt_1 = img_t - current_time;
                    double dt_2 = t - img_t;
                    current_time = img_t;
                    assert(dt_1 >= 0);
                    assert(dt_2 >= 0);
                    assert(dt_1 + dt_2 > 0);
                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x();
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y();
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z();
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x();
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y();
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z();
                    estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                }
            }

            // TicToc t_s;
            map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
            // 提取当前帧所有特征点及对应的图像帧信息
            for (unsigned int i = 0; i < img_msg->points.size(); i++) {
                int v = img_msg->id_of_point[i] + 0.5;
                int feature_id = v / NUM_OF_CAM;
                int camera_id = v % NUM_OF_CAM;
                double x = img_msg->points[i].x();
                double y = img_msg->points[i].y();
                double z = img_msg->points[i].z();
                double p_u = img_msg->u_of_point[i];
                double p_v = img_msg->v_of_point[i];
                double velocity_x = img_msg->velocity_x_of_point[i];
                double velocity_y = img_msg->velocity_y_of_point[i];
                assert(z == 1);
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                image[feature_id].emplace_back(camera_id, xyz_uv_velocity);
            }
            TicToc t_processImage;
            /// 调用估计器根据图像信息进行处理,重要函数
            estimator.processImage(image, img_msg->header);
            // 如果已经进入全局优化阶段,则添加显示轨迹信息和输出信息
            if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR) {
                Vector3d p_wi;
                Quaterniond q_wi;
                q_wi = Quaterniond(estimator.Rs[WINDOW_SIZE]);
                p_wi = estimator.Ps[WINDOW_SIZE];
                vPath_to_draw.push_back(p_wi);
                double dStamp = estimator.Headers[WINDOW_SIZE];
                cout << "1 BackEnd processImage dt: " << fixed
                     << t_processImage.toc() << " stamp: "
                     <<  dStamp << " p_wi: " << p_wi.transpose() << endl;
                ofs_pose << fixed << dStamp << " " << p_wi.transpose()
                         << " " << q_wi.coeffs().transpose() << endl;
            }
        }
        m_estimator.unlock();
    }
}

void System::draw() {
    // 创建pangolin绘制轨迹窗口
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    s_cam = pangolin::OpenGlRenderState(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 384, 0.1, 1000),
            pangolin::ModelViewLookAt(-5, 0, 15, 7, 0, 0, 1.0, 0.0, 0.0)
    );

    d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(0.75f, 0.75f, 0.75f, 0.75f);
        glColor3f(0, 0, 1);
        pangolin::glDrawAxis(3);

        // 绘制轨迹的位置信息
        glColor3f(0, 0, 0);
        glLineWidth(2);
        glBegin(GL_LINES);
        int nPath_size = vPath_to_draw.size();
        for(int i = 0; i < nPath_size-1; ++i) {
            glVertex3f(vPath_to_draw[i].x(), vPath_to_draw[i].y(), vPath_to_draw[i].z());
            glVertex3f(vPath_to_draw[i+1].x(), vPath_to_draw[i+1].y(), vPath_to_draw[i+1].z());
        }
        glEnd();

        // 绘制非线性优化的窗口信息
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR) {
            glPointSize(5);
            glBegin(GL_POINTS);
            for(int i = 0; i < WINDOW_SIZE+1;++i) {
                Vector3d p_wi = estimator.Ps[i];
                glColor3f(1, 0, 0);
                glVertex3d(p_wi[0],p_wi[1],p_wi[2]);
            }
            glEnd();
        }
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
}