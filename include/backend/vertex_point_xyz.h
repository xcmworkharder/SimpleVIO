#ifndef SIMPLE_VIO_VERTEX_POINT_XYZ_H
#define SIMPLE_VIO_VERTEX_POINT_XYZ_H

#include "vertex.h"

namespace myslam {
namespace backend {

/**
* @brief 以xyz形式参数化的顶点
*/
class VertexPointXYZ : public Vertex {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VertexPointXYZ() : Vertex(3) {}

    std::string typeInfo() const { return "VertexPointXYZ"; }
};

}
}

#endif //SIMPLE_VIO_VERTEX_POINT_XYZ_H
