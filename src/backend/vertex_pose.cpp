#include "backend/vertex_pose.h"
#include "../third_party/Sophus/sophus/se3.hpp"

namespace myslam {
namespace backend {

void VertexPose::plus(const VecX& delta) {
    VecX& the_parameters = parameters();
    the_parameters.head<3>() += delta.head<3>();
    Qd q(the_parameters[6], the_parameters[3], the_parameters[4], the_parameters[5]);
    q = q * Sophus::SO3d::exp(Vec3(delta[3], delta[4], delta[5])).unit_quaternion();  // right multiplication with so3
    q.normalized();
    the_parameters[3] = q.x();
    the_parameters[4] = q.y();
    the_parameters[5] = q.z();
    the_parameters[6] = q.w();
}

}
}
