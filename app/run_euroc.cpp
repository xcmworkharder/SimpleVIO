#include <iostream>
#include <unistd.h>
#include <fstream>
#include <Eigen/Dense>
#include <thread>
#include "cv.h"
#include <opencv2/opencv.hpp>
#include "system.h"


using namespace std;
using namespace Eigen;
using namespace cv;

const int nDelayTimes = 2;
// 相机采集图像数据
string sData_path = "/home/xcmworkharder/Documents/forlearning/SLam Learning/DataSet/mav0/";
// 各类配置信息
string sConfig_path = "../config/";

// vio主系统类智能指针
shared_ptr<System> pSystem;

// 提取Imu信息线程函数
void pubImuData() {
    string sImu_data_file = sConfig_path + "MH_05_imu0.txt"; // 周期5毫秒
    cout << "1: Start PubImuData, sImu_data_file: " << sImu_data_file << endl;
    ifstream fsImu;
    fsImu.open(sImu_data_file.c_str());
    if (!fsImu.is_open()) {
        cerr << "Failed to open imu file! " << sImu_data_file << endl;
        return;
    }

    string sImu_line;
    double dStampNSec = 0.0;
    Vector3d vAcc;
    Vector3d vGyro;
    while(getline(fsImu, sImu_line) && !sImu_line.empty()) {
        istringstream ssImuData(sImu_line);
        ssImuData >> dStampNSec >> vGyro.x() >> vGyro.y() >> vGyro.z()
                                >> vAcc.x() >> vAcc.y() >> vAcc.z();
        // 发布imu信息
        pSystem->pubImuData(dStampNSec / 1e9, vGyro, vAcc); // 时间戳统一到秒
        // 控制发布频率, 5000*2为10毫秒,即频率100HZ
        usleep(5000 * nDelayTimes);
    }
    fsImu.close();
}

// 提取发布图像信息线程
void pubImageData() {
    string sImage_file = sConfig_path + "MH_05_cam0.txt"; // 周期为50毫秒
    cout << "1: Start pubImageData, sImage_file: " << sImage_file << endl;
    ifstream fsImage;
    fsImage.open(sImage_file.c_str());
    if (!fsImage.is_open()) {
        cerr << "Failed to open image file! " << sImage_file << endl;
        return;
    }
    string sImage_line;
    double dStampNSec;
    string sImgFileName;

    while(getline(fsImage, sImage_line) && !sImage_line.empty()) {
        istringstream ssImageData(sImage_line);
        ssImageData >> dStampNSec >> sImgFileName;
        string imagePath = sData_path + "cam0/data/" + sImgFileName;
        // 提取图像信息
        Mat img = imread(imagePath.c_str(), 0); // 单通道
        if(img.empty()) {
            cerr << "image is empty! path: " << imagePath << endl;
            return;
        }
        // 发布图像信息, 50000*2=100000, 100毫秒,相当于10Hz
        pSystem->pubImageData(dStampNSec / 1e9, img);
        usleep(50000 * nDelayTimes);
    }
    fsImage.close();
}

int main(int argc, char** argv) {

    // 设置并启动vio系统
    pSystem.reset(new System(sConfig_path));

    // 启动信息发布,vio估计,绘图线程
    thread thd_BackEnd(&System::processBackEnd, pSystem);
    thread thd_pubImuData(pubImuData);
    thread thd_pubImageData(pubImageData);
    thread thd_Draw(&System::draw, pSystem);

    // 等待线程结束
    thd_pubImuData.join();
    thd_pubImageData.join();
    thd_BackEnd.join();
    thd_Draw.join();

    cout << "main end... see you... " << endl;
    return 0;
}