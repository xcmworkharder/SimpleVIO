#ifndef SIMPLE_VIO_INTEGRATION_BASE_H
#define SIMPLE_VIO_INTEGRATION_BASE_H

#include <vector>
#include "utility/utility.h"
#include "parameters.h"

using namespace Eigen;
using namespace std;

/**
 * 预积分类
 */
class IntegrationBase {
public:
    IntegrationBase() = delete; // 禁止默认构造函数
    IntegrationBase(const Vector3d& _acc_0, const Vector3d& _gyro_0,
                    const Vector3d& _linearized_ba,
                    const Vector3d& _linearized_bg)
            : acc_0{_acc_0}, gyro_0{_gyro_0},   // 用{}初始化进行统一,防止初始化和函数调用混淆,()也可以
              linearized_acc{_acc_0}, linearized_gyro{_gyro_0},
              linearized_ba{_linearized_ba}, linearized_bg{_linearized_bg},
              jacobian{Matrix<double, 15, 15>::Identity()},
              covariance{Matrix<double, 15, 15>::Zero()},
              sum_dt{0.0}, delta_p{Vector3d::Zero()},
              delta_q{Quaterniond::Identity()}, delta_v{Vector3d::Zero()} {
        // 对噪声阵进行分块初始化
        noise = Matrix<double, 18, 18>::Zero();
        noise.block<3, 3>(0, 0) = (ACC_N * ACC_N) * Matrix3d::Identity();
        noise.block<3, 3>(3, 3) = (GYR_N * GYR_N) * Matrix3d::Identity();
        noise.block<3, 3>(6, 6) = (ACC_N * ACC_N) * Matrix3d::Identity();
        noise.block<3, 3>(9, 9) = (GYR_N * GYR_N) * Matrix3d::Identity();
        noise.block<3, 3>(12, 12) = (ACC_N * ACC_N) * Matrix3d::Identity();
        noise.block<3, 3>(15, 15) = (GYR_N * GYR_N) * Matrix3d::Identity();
    }

    // 数据放入缓存
    void push_back(double dt, const Vector3d& acc, const Vector3d& gyro) {
        dt_buf.push_back(dt);
        acc_buf.emplace_back(acc);
        gyro_buf.emplace_back(gyro);
        propagate(dt, acc, gyro);
    }

    // 信息及误差传播
    void propagate(double _dt, const Vector3d& _acc_1, const Vector3d& _gyro_1) {
        dt = _dt;
        acc_1 = _acc_1;
        gyro_1 = _gyro_1;
        Vector3d result_delta_p;
        Quaterniond result_delta_q;
        Vector3d result_delta_v;
        Vector3d result_linearized_ba;
        Vector3d result_linearized_bg;

        midPointIntegration(_dt, acc_0, gyro_0, _acc_1, _gyro_1, delta_p,
                            delta_q, delta_v, linearized_ba, linearized_bg,
                            result_delta_p, result_delta_q, result_delta_v,
                            result_linearized_ba, result_linearized_bg, true);
        delta_p = result_delta_p;
        delta_q = result_delta_q;
        delta_v = result_delta_v;
        linearized_ba = result_linearized_ba;
        linearized_bg = result_linearized_bg;
        delta_q.normalize();
        sum_dt += dt;
        acc_0 = acc_1;
        gyro_0 = gyro_1;
    }

    // 中值法进行预积分处理
    void midPointIntegration(double _dt,
                             const Vector3d& _acc_0, const Vector3d& _gyro_0,
                             const Vector3d& _acc_1, const Vector3d& _gyro_1,
                             const Vector3d& delta_p, const Quaterniond& delta_q,
                             const Vector3d& delta_v, const Vector3d& linearized_ba,
                             const Vector3d& linearized_bg, Vector3d& result_delta_p,
                             Quaterniond& result_delta_q, Vector3d& result_delta_v,
                             Vector3d& result_linearized_ba, Vector3d& result_linearized_bg,
                             bool updata_jacobian) {
        Vector3d un_acc_0 = delta_q * (_acc_0 - linearized_ba);
        Vector3d un_gyro = 0.5 * (_gyro_0 + _gyro_1) - linearized_bg;
        result_delta_q = delta_q * Quaterniond(1, un_gyro(0) * dt / 2,
                                                  un_gyro(1) * dt / 2,
                                                  un_gyro(2) * dt / 2);
        Vector3d un_acc_1 = result_delta_q * (_acc_1 - linearized_ba);
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        result_delta_p = delta_p + delta_v * _dt + 0.5 * un_acc * _dt * _dt;
        result_delta_v = delta_v + un_acc * _dt;
        result_linearized_ba = linearized_ba;
        result_linearized_bg = linearized_bg;

        // 更新雅克比矩阵和噪声
        if (updata_jacobian) {
            Vector3d w_x = 0.5 * (_gyro_0 + _gyro_1) - linearized_bg;
            Vector3d a_0_x = _acc_0 - linearized_ba;
            Vector3d a_1_x = _acc_1 - linearized_ba;
            Matrix3d R_w_x, R_a_0_x, R_a_1_x;

            R_w_x = Utility::skewSymmetric(w_x);
            R_a_0_x = Utility::skewSymmetric(a_0_x);
            R_a_1_x = Utility::skewSymmetric(a_1_x);

            // 转移矩阵
            MatrixXd F = MatrixXd::Zero(15, 15);
            F.block<3, 3>(0, 0) = Matrix3d::Identity();
            F.block<3, 3>(0, 3) = -0.25 * delta_q.toRotationMatrix()
                                  * R_a_0_x * _dt * _dt - 0.25 * result_delta_q.toRotationMatrix()
                                  * R_a_1_x * (Matrix3d::Identity() - R_w_x * _dt) * _dt * _dt;
            F.block<3, 3>(0, 6) = MatrixXd::Identity(3,3) * _dt;
            F.block<3, 3>(0, 9) = -0.25 * (delta_q.toRotationMatrix()
                                           + result_delta_q.toRotationMatrix()) * _dt * _dt;
            F.block<3, 3>(0, 12) = -0.25 * result_delta_q.toRotationMatrix()
                                   * R_a_1_x * _dt * _dt * -_dt;
            F.block<3, 3>(3, 3) = Matrix3d::Identity() - R_w_x * _dt;
            F.block<3, 3>(3, 12) = -1.0 * MatrixXd::Identity(3,3) * _dt;
            F.block<3, 3>(6, 3) = -0.5 * delta_q.toRotationMatrix() * R_a_0_x * _dt
                                  - 0.5 * result_delta_q.toRotationMatrix()
                                    * R_a_1_x * (Matrix3d::Identity() - R_w_x * _dt) * _dt;
            F.block<3, 3>(6, 6) = Matrix3d::Identity();
            F.block<3, 3>(6, 9) = -0.5 * (delta_q.toRotationMatrix()
                                          + result_delta_q.toRotationMatrix()) * _dt;
            F.block<3, 3>(6, 12) = -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * (-_dt);
            F.block<3, 3>(9, 9) = Matrix3d::Identity();
            F.block<3, 3>(12, 12) = Matrix3d::Identity();

            // 噪声转换矩阵
            MatrixXd V = MatrixXd::Zero(15,18);
            V.block<3, 3>(0, 0) =  0.25 * delta_q.toRotationMatrix() * _dt * _dt;
            V.block<3, 3>(0, 3) =  0.25 * -result_delta_q.toRotationMatrix()
                                   * R_a_1_x  * _dt * _dt * 0.5 * _dt;
            V.block<3, 3>(0, 6) =  0.25 * result_delta_q.toRotationMatrix() * _dt * _dt;
            V.block<3, 3>(0, 9) =  V.block<3, 3>(0, 3);
            V.block<3, 3>(3, 3) =  0.5 * MatrixXd::Identity(3,3) * _dt;
            V.block<3, 3>(3, 9) =  0.5 * MatrixXd::Identity(3,3) * _dt;
            V.block<3, 3>(6, 0) =  0.5 * delta_q.toRotationMatrix() * _dt;
            V.block<3, 3>(6, 3) =  0.5 * -result_delta_q.toRotationMatrix()
                                   * R_a_1_x  * _dt * 0.5 * _dt;
            V.block<3, 3>(6, 6) =  0.5 * result_delta_q.toRotationMatrix() * _dt;
            V.block<3, 3>(6, 9) =  V.block<3, 3>(6, 3);
            V.block<3, 3>(9, 12) = MatrixXd::Identity(3,3) * _dt;
            V.block<3, 3>(12, 15) = MatrixXd::Identity(3,3) * _dt;

            //step_jacobian = F;
            //step_V = V;
            jacobian = F * jacobian;
            covariance = F * covariance * F.transpose() + V * noise * V.transpose();
        }
    }

    // 利用在线估计的bias,进行重新预积分计算
    void repropagate(const Vector3d& _linearized_ba, const Vector3d& _linearized_bg) {
        sum_dt = 0.0;
        acc_0 = linearized_acc;
        gyro_0 = linearized_gyro;
        delta_p.setZero();
        delta_q.setIdentity();
        delta_v.setZero();
        linearized_ba = _linearized_ba;
        linearized_bg = _linearized_bg;
        jacobian.setIdentity();
        covariance.setZero();
        for (int i = 0; i < static_cast<int>(dt_buf.size()); ++i) {
            propagate(dt_buf[i], acc_buf[i], gyro_buf[i]);
        }
    }

    Eigen::Matrix<double, 15, 1> evaluate(const Eigen::Vector3d &Pi,
                                          const Eigen::Quaterniond &Qi,
                                          const Eigen::Vector3d &Vi,
                                          const Eigen::Vector3d &Bai,
                                          const Eigen::Vector3d &Bgi,
                                          const Eigen::Vector3d &Pj,
                                          const Eigen::Quaterniond &Qj,
                                          const Eigen::Vector3d &Vj,
                                          const Eigen::Vector3d &Baj,
                                          const Eigen::Vector3d &Bgj) {
        Eigen::Matrix<double, 15, 1> residuals;

        Eigen::Matrix3d dp_dba = jacobian.block<3, 3>(O_P, O_BA);
        Eigen::Matrix3d dp_dbg = jacobian.block<3, 3>(O_P, O_BG);

        Eigen::Matrix3d dq_dbg = jacobian.block<3, 3>(O_R, O_BG);

        Eigen::Matrix3d dv_dba = jacobian.block<3, 3>(O_V, O_BA);
        Eigen::Matrix3d dv_dbg = jacobian.block<3, 3>(O_V, O_BG);

        Eigen::Vector3d dba = Bai - linearized_ba;
        Eigen::Vector3d dbg = Bgi - linearized_bg;

        Eigen::Quaterniond corrected_delta_q = delta_q * Utility::deltaQ(dq_dbg * dbg);
        Eigen::Vector3d corrected_delta_v = delta_v + dv_dba * dba + dv_dbg * dbg;
        Eigen::Vector3d corrected_delta_p = delta_p + dp_dba * dba + dp_dbg * dbg;

        residuals.block<3, 1>(O_P, 0) = Qi.inverse()
                                        * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt)
                                        - corrected_delta_p;
        residuals.block<3, 1>(O_R, 0) = 2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
        residuals.block<3, 1>(O_V, 0) = Qi.inverse() * (G * sum_dt + Vj - Vi) - corrected_delta_v;
        residuals.block<3, 1>(O_BA, 0) = Baj - Bai;
        residuals.block<3, 1>(O_BG, 0) = Bgj - Bgi;
        return residuals;
    }

    double dt; // 积分间隔
    Vector3d acc_0, gyro_0; // 一段区间加速度计和陀螺的开始点数值
    Vector3d acc_1, gyro_1; // 一段区间加速度计和陀螺的结束点数值
    const Vector3d linearized_acc, linearized_gyro; // 初始化后保持不变
    Vector3d linearized_ba, linearized_bg;          // 加速度计和陀螺的bias
    Matrix<double, 15, 15> jacobian, covariance;    // 15维包括速度\位置\角度预积分及加速度计和陀螺bias
    Matrix<double, 15, 15> step_jacobian;
    Matrix<double, 15, 18> step_V;
    Matrix<double, 18, 18> noise;                   // 18维

    double sum_dt;
    Vector3d delta_p;
    Quaterniond delta_q;
    Vector3d delta_v;

    vector<double> dt_buf;
    vector<Vector3d> acc_buf;
    vector<Vector3d> gyro_buf;
};


#endif //SIMPLE_VIO_INTEGRATION_BASE_H
