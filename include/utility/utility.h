#ifndef SIMPLE_VIO_UTILITY_H
#define SIMPLE_VIO_UTILITY_H

#include <cmath>
#include <cstring>
#include <Eigen/Dense>

/**
 * 提供一些基础四元数计算函数
 */
class Utility {
public:
    /**
     * q(b_i, b_k+1) = q(b_i, b_k) * [1 1/2w].transpose();
     * @tparam Derived data_type double, float...
     * @param theta w
     * @return [1 1/2w].transpose()
     */
    template <typename Derived>
    static Eigen::Quaternion<typename Derived::Scalar> deltaQ(
            const Eigen::MatrixBase<Derived>& theta) {
        typedef typename Derived::Scalar Scalar_t;

        Eigen::Quaternion<Scalar_t> dq;
        Eigen::Matrix<Scalar_t, 3, 1> half_theta = theta;
        half_theta /= static_cast<Scalar_t>(2.0);
        dq.w() = static_cast<Scalar_t>(1.0);
        dq.x() = half_theta.x();
        dq.y() = half_theta.y();
        dq.z() = half_theta.z();
        return dq;
    }

    // 使用q构造反对称阵
    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 3, 3> skewSymmetric(
            const Eigen::MatrixBase<Derived>& q) {
        typedef typename Derived::Scalar Scalar_t;

        Eigen::Matrix<Scalar_t, 3, 3> res;
        res << static_cast<Scalar_t>(0), -q(2), q(1),
                q(2), static_cast<Scalar_t>(0), -q(0),
                -q(1), q(0), static_cast<Scalar_t>(0);
        return res;
    }

    // 返回四元数的正值
    template <typename Derived>
    static Eigen::Quaternion<typename Derived::Scalar> positify(
            const Eigen::QuaternionBase<Derived>& q) {
//        Eigen::Quaternion<typename Derived::Scalar> p(-q.w(), -q.x(), -q.y(), -q.z());
//        return q.template w() >= (typename Derived::Scalar)(0.0) ?
//               q : Eigen::Quaternion<typename Derived::Scalar>(-q.w(), -q.x(), -q.y(), -q.z());
        return q;
    }

    // 返回四元数的左乘矩阵
    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 4, 4> qLeft(
            const Eigen::QuaternionBase<Derived>& q) {
        Eigen::Quaternion<typename Derived::Scalar> qq = positify(q);
        Eigen::Matrix<typename Derived::Scalar, 4, 4> res;
        res(0, 0) = qq.w();
        res.template block<1, 3>(0, 1) = -qq.vec().transpose();
        res.template block<3, 1>(1, 0) = qq.vec();
        res.template block<3, 3>(1, 1) =
                qq.w() * Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity()
                + skewSymmetric(qq.vec());
        return res;
    }

    // 返回四元数的右乘矩阵
    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 4, 4> qRight(
            const Eigen::QuaternionBase<Derived>& p) {
        Eigen::Quaternion<typename Derived::Scalar> pp = positify(p);
        Eigen::Matrix<typename Derived::Scalar, 4, 4> res;
        res(0, 0) = pp.w();
        res.template block<1, 3>(0, 1) = -pp.vec().transpose();
        res.template block<3, 1>(1, 0) = pp.vec();
        res.template block<3, 3>(1, 1) =
                pp.w() * Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity()
                - skewSymmetric(pp.vec());
        return res;
    }

    // 旋转矩阵向ypr欧拉角转化
    static Eigen::Vector3d R2ypr(const Eigen::Matrix3d& R) {
        Eigen::Vector3d n = R.col(0);
        Eigen::Vector3d o = R.col(1);
        Eigen::Vector3d a = R.col(2);

        Eigen::Vector3d ypr(3);
        double y = atan2(n(1), n(0));
        double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
        double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
        ypr(0) = y;
        ypr(1) = p;
        ypr(2) = r;

        return ypr / M_PI * 180.0;
    }

    // ypr向旋转矩阵R转换
    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 3, 3> ypr2R(
            const Eigen::MatrixBase<Derived>& ypr) {
        typedef typename Derived::Scalar Scalar_t;

        Scalar_t y = ypr(0) / 180.0 * M_PI;
        Scalar_t p = ypr(1) / 180.0 * M_PI;
        Scalar_t r = ypr(2) / 180.0 * M_PI;

        Eigen::Matrix<Scalar_t, 3, 3> Rz;
        Rz << cos(y), -sin(y), 0,
                sin(y), cos(y), 0,
                0, 0, 1;

        Eigen::Matrix<Scalar_t, 3, 3> Ry;
        Ry << cos(p), 0., sin(p),
                0., 1., 0.,
                -sin(p), 0., cos(p);

        Eigen::Matrix<Scalar_t, 3, 3> Rx;
        Rx << 1., 0., 0.,
                0., cos(r), -sin(r),
                0., sin(r), cos(r);

        return Rz * Ry * Rx;
    }

    // 根据重力矢量转换为相机到世界系的旋转矩阵
    static Eigen::Matrix3d g2R(const Eigen::Vector3d& g) {
        Eigen::Matrix3d R0;
        Eigen::Vector3d ng1 = g.normalized();
        Eigen::Vector3d ng2{0, 0, 1.0};
        R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();
        double yaw = R2ypr(R0).x();
        R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
        return R0;
    }

    // 单位结构体
    template <size_t N>
    struct uint_ {
    };

    // 递归进行迭代输出
    template <size_t N, typename Lambda, typename IterT>
    void unroller(const Lambda& f, const IterT& iter, uint_<N>) {
        unroller(f, iter, uint_<N - 1>());
        f(iter + N);
    }

    // 迭代输出
    template <typename Lambda, typename IterT>
    void unroller(const Lambda& f, const IterT& iter, uint_<0>)
    {
        f(iter);
    }

    // 角度统一到[-180, 180]范围内
    template <typename T>
    static T normalizeAngle(const T& angle_degrees) {
        T two_pi(2.0 * 180);
        if (angle_degrees > 0)
            return angle_degrees -
                   two_pi * std::floor((angle_degrees + T(180)) / two_pi);
        else
            return angle_degrees +
                   two_pi * std::floor((-angle_degrees + T(180)) / two_pi);
    }
};

#endif //SIMPLE_VIO_UTILITY_H
