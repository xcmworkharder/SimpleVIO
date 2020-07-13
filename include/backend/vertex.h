#ifndef SIMPLE_VIO_VERTEX_H
#define SIMPLE_VIO_VERTEX_H

#include "eigen_types.h"

namespace myslam {
namespace backend {
extern unsigned long global_vertex_id; // 供外部使用
/**
* @brief 顶点，对应一个parameter block
* 变量值以VecX存储，需要在构造时指定维度
*/
class Vertex {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    /**
     * 构造函数
     * @param num_dimension 顶点自身维度
     * @param local_dimension 本地参数化维度，默认为-1,表示与本身维度一样
     */
    explicit Vertex(int num_dimension, int local_dimension = -1);

    virtual ~Vertex();

    /// 返回变量维度
    int dimension() const;

    /// 返回变量本地维度
    int localDimension() const;

    /// 该顶点的id
    unsigned long id() const { return id_; }

    /// 返回参数值
    VecX parameters() const { return parameters_; }

    /// 返回参数值的引用
    VecX& parameters() { return parameters_; }

    /// 设置参数值
    void setParameters(const VecX& params) { parameters_ = params; }

    // 备份和回滚参数，用于丢弃一些迭代过程中不好的估计
    void backUpParameters() { parameters_backup_ = parameters_; }

    void rollBackParameters() { parameters_ = parameters_backup_; }

    /// 加法，可重定义
    /// 默认是向量加
    virtual void plus(const VecX& delta);

    /// 返回顶点的名称，在子类中实现
    virtual std::string typeInfo() const = 0;

    int orderingId() const { return ordering_id_; }

    void setOrderingId(unsigned long id) { ordering_id_ = id; };

    /// 固定该点的估计值
    void setFixed(bool fixed = true) {
        fixed_ = fixed;
    }

    /// 测试该点是否被固定
    bool isFixed() const { return fixed_; }

protected:
    VecX parameters_;           // 实际存储的变量值
    VecX parameters_backup_;    // 每次迭代优化中对参数进行备份，用于回滚
    int local_dimension_;       // 局部参数化维度
    unsigned long id_;          // 顶点的id，自动生成

    /// ordering id是在problem中排序后的id，用于寻找雅可比对应块
    /// ordering id带有维度信息，例如ordering_id=6则对应Hessian中的第6列
    /// 从零开始
    unsigned long ordering_id_ = 0;
    bool fixed_ = false;    // 是否固定
};

}
}

#endif //SIMPLE_VIO_VERTEX_H
