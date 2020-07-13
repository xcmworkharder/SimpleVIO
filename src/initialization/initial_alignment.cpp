#include "initialization/initial_alignment.h"
#include <iostream>

/**
 * 陀螺偏置校正
 *    根据视觉SFM的结果来校正陀螺仪Bias
 *       将相邻帧之间SFM求解出来的旋转矩阵与IMU预积分的旋转量对齐
 *       得到了新的Bias后,需要对IMU预积分进行重新计算,调用repropagate
 * @param all_image_frame
 * @param bgs
 */
void solveGyroscopeBias(map<double, ImageFrame>& all_image_frame, Vector3d* bgs) {
    Matrix3d A;
    Vector3d b;
    Vector3d delta_bg;
    A.setZero();
    b.setZero();
    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    for (frame_i = all_image_frame.begin();
         next(frame_i) != all_image_frame.end(); ++frame_i) {
        frame_j = next(frame_i);
        Matrix3d tmp_A;
        tmp_A.setZero();
        VectorXd tmp_b(3);
        tmp_b.setZero();

        // R_ij = (R^c0_bk)^-1 * (R^c0_bk+1)
        // 转换为四元数 q_ij = (q^c0_bk)^-1 * (q^c0_bk+1)
        Quaterniond q_ij(frame_i->second.R.transpose() * frame_j->second.R);
        // tmp_A = J_j_bw
        tmp_A = frame_j->second.pre_integration->jacobian.block<3, 3>(O_R, O_BG);
        // tmp_b = 2 * (r^bk_bk+1)^-1 * (q^c0_bk)^-1 * (q^c0_bk+1)
        //      = 2 * (r^bk_bk+1)^-1 * q_ij
        tmp_b = 2 * (frame_j->second.pre_integration->delta_q.inverse() * q_ij).vec();
        // tmp_A * delta_bg = tmp_b
        A += tmp_A.transpose() * tmp_A;
        b += tmp_A.transpose() * tmp_b;
    }

    // LDLT法求解
    delta_bg = A.ldlt().solve(b);

    for (int i = 0; i <= WINDOW_SIZE; ++i) {
        bgs[i] += delta_bg;
    }

    // 使用优化得出的bias进行重新预积分
    for (frame_i = all_image_frame.begin();
         next(frame_i) != all_image_frame.end(); ++frame_i) {
        frame_j = next(frame_i);
        frame_j->second.pre_integration->repropagate(Vector3d::Zero(), bgs[0]);
    }
}

/**
 * 找到重力矢量切面的一对正交基,将三维向量自由度变为2
 * g = ||g||gn + w1b1 + w2b2, gn为重力归一化, w1,w2为待优化变量
 * @param g0
 * @return
 */
MatrixXd tangentBasis(Vector3d& g0) {
    Vector3d b, c;
    Vector3d a = g0.normalized();
    Vector3d tmp(0, 0, 1);
    if (a == tmp) { // 这里a == tmp编译有问题提示,功能不受影响
        tmp << 1, 0, 0;
    }
    b = (tmp - a * (a.transpose() * tmp)).normalized();
    c = a.cross(b);
    MatrixXd bc(3, 2);
    bc.block<3, 1>(0, 0) = b;
    bc.block<3, 1>(0, 1) = c;
    return bc;
}

/**
 * 优化重力矢量,得到C0系下重力矢量
 * 将其旋转到惯性坐标系的Z轴方向(0, 0, g),得出相机系到惯性系的转换矩阵
 * @param all_image_frame
 * @param g
 * @param x
 */
void refineGravity(map<double, ImageFrame>& all_image_frame,
                   Vector3d& g, VectorXd& x) {

    Vector3d g0 = g.normalized() * G.norm();
    Vector3d lx, ly;
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 2 + 1; // 3表示每一帧有3维速度

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;

    for (int k = 0; k < 4; ++k) { //迭代4次
        //lxly = b = [b1,b2]
        MatrixXd lxly(3, 2);
        lxly = tangentBasis(g0);
        int i = 0;
        for (frame_i = all_image_frame.begin();
             next(frame_i) != all_image_frame.end(); ++frame_i, ++i) {
            frame_j = next(frame_i);

            MatrixXd tmp_A(6, 9);
            tmp_A.setZero();
            VectorXd tmp_b(6);
            tmp_b.setZero();

            double dt = frame_j->second.pre_integration->sum_dt;

            // tmp_A(6,9) = [-I*dt           0             (R^bk_c0)*dt*dt*b/2   (R^bk_c0)*((p^c0_ck+1)-(p^c0_ck))  ]
            //              [ -I    (R^bk_c0)*(R^c0_bk+1)      (R^bk_c0)*dt*b                  0                    ]
            // tmp_b(6,1) = [ (a^bk_bk+1)+(R^bk_c0)*(R^c0_bk+1)*p^b_c-p^b_c - (R^bk_c0)*dt*dt*||g||*(g^-)/2 , (b^bk_bk+1)-(R^bk_c0)dt*||g||*(g^-)]^T
            // tmp_A * x = tmp_b 求解最小二乘问题
            tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
            tmp_A.block<3, 2>(0, 6) = frame_i->second.R.transpose()
                                      * dt * dt / 2 * Matrix3d::Identity() * lxly;
            tmp_A.block<3, 1>(0, 8) = frame_i->second.R.transpose()
                                      * (frame_j->second.T - frame_i->second.T) / 100.0; // /100.0防止数值不平衡
            tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p
                                      + frame_i->second.R.transpose() * frame_j->second.R
                                        * TIC[0] - TIC[0]
                                      - frame_i->second.R.transpose() * dt * dt / 2 * g0;

            tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
            tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
            tmp_A.block<3, 2>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity() * lxly;
            tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v
                                      - frame_i->second.R.transpose() * dt * Matrix3d::Identity() * g0;

            Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
            //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
            //MatrixXd cov_inv = cov.inverse();
            cov_inv.setIdentity();

            MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A; // dim:9*9
            VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b; // dim:9*1

            /**
             * 变成分块稀疏形式,视觉速度部分沿着图像帧计算,重力2分量和尺度s累积计算
             */
            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
            b.tail<3>() += r_b.tail<3>();

            A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
            A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();

        }
        A = A * 1000.0; // * 1000便于计算稳定
        b = b * 1000.0;
        x = A.ldlt().solve(b);
        //dg = [w1,w2]^T
        VectorXd dg = x.segment<2>(n_state - 3);
        g0 = (g0 + lxly * dg).normalized() * G.norm(); // 迭代计算
        //double s = x(n_state - 1);
    }
    g = g0;
}

/**
 * 估计速度,重力加速度和尺度等参数, 后续调用上面的refineGravity,利用重力自身约束进行优化修正
*/
bool linearAlignment(map<double, ImageFrame>& all_image_frame,
                     Vector3d&g, VectorXd& x) {

    int all_frame_count = all_image_frame.size();
    //优化量x的总维度
    int n_state = all_frame_count * 3 + 3 + 1; // 此处重力加速度为3维

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    int i = 0;
    for (frame_i = all_image_frame.begin();
         next(frame_i) != all_image_frame.end(); ++frame_i, ++i) {
        frame_j = next(frame_i);

        MatrixXd tmp_A(6, 10);
        tmp_A.setZero();
        VectorXd tmp_b(6);
        tmp_b.setZero();

        double dt = frame_j->second.pre_integration->sum_dt;

        // tmp_A(6,10) = H^bk_bk+1 = [-I*dt           0             (R^bk_c0)*dt*dt/2   (R^bk_c0)*((p^c0_ck+1)-(p^c0_ck))  ]
        //                           [ -I    (R^bk_c0)*(R^c0_bk+1)      (R^bk_c0)*dt                  0                    ]
        // tmp_b(6,1 ) = z^bk_bk+1 = [ (a^bk_bk+1)+(R^bk_c0)*(R^c0_bk+1)*p^b_c-p^b_c , (b^bk_bk+1)]^T
        // tmp_A * x = tmp_b 求解最小二乘问题
        tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
        tmp_A.block<3, 3>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity();
        tmp_A.block<3, 1>(0, 9) = frame_i->second.R.transpose()
                                  * (frame_j->second.T - frame_i->second.T) / 100.0;
        tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p
                                  + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0];
        //cout << "delta_p   " << frame_j->second.pre_integration->delta_p.transpose() << endl;
        tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
        tmp_A.block<3, 3>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity();
        tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v;
        //cout << "delta_v   " << frame_j->second.pre_integration->delta_v.transpose() << endl;

        Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
        //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
        //MatrixXd cov_inv = cov.inverse();
        cov_inv.setIdentity();

        MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
        VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
        b.segment<6>(i * 3) += r_b.head<6>();

        A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
        b.tail<4>() += r_b.tail<4>();

        A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
        A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
    }
    A = A * 1000.0;
    b = b * 1000.0;
    x = A.ldlt().solve(b);

    double s = x(n_state - 1) / 100.0;
    g = x.segment<3>(n_state - 4);

    if (fabs(g.norm() - G.norm()) > 1.0 || s < 0) {
        return false;
    }

    // 重力优化修正
    refineGravity(all_image_frame, g, x);

    s = (x.tail<1>())(0) / 100.0;
    (x.tail<1>())(0) = s;

    if (s < 0.0) {
        return false;
    } else {
        return true;
    }
}

// 视觉和IMU对准
bool visualIMUAlignment(map<double, ImageFrame>& all_image_frame,
                        Vector3d* bgs, Vector3d& g, VectorXd& x) {
    // 陀螺bias校正
    solveGyroscopeBias(all_image_frame, bgs);
    // 初始化速度,重力,尺度因子
    if (linearAlignment(all_image_frame, g, x)) {
        return true;
    } else {
        return false;
    }
}