#include <iostream>
#include "utility/utility.h"
//#include "initialization/initial_sfm.h"

using namespace std;
using namespace Eigen;

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

int main(int argc, char** argv) {


    Vector3d a{1, 2, 3};
    Vector3d b(1, 2, 3);
    if (a == b) {
        cout << "a equals b" << endl;
    }
    Matrix3d res = skewSymmetric(a);
    cout << res << endl;
    return 0;
}
