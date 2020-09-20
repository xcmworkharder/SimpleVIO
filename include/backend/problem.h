#ifndef SIMPLE_VIO_PROBLEM_H
#define SIMPLE_VIO_PROBLEM_H

#include <unordered_map>
#include <map>
#include <memory>
#include <thread>
#include <mutex>

#include "eigen_types.h"
#include "edge.h"
#include "vertex.h"

namespace myslam {
namespace backend {
typedef unsigned long ulong;
// 注意: 这里如果使用unordered_map,vertex顺序就是乱的,估计的结果也是乱的
//typedef std::unordered_map<ulong, std::shared_ptr<Vertex>> HashVertex;
typedef std::map<ulong, std::shared_ptr<Vertex>> HashVertex;
typedef std::unordered_map<ulong, std::shared_ptr<Edge>> HashEdge;
typedef std::unordered_multimap<ulong, std::shared_ptr<Edge>> HashVertexIdToEdge;

class Problem {
public:

    /**
     * 问题的类型
     * SLAM问题还是通用的问题
     *
     * 如果是SLAM问题那么pose和landmark是区分开的，Hessian以稀疏方式存储
     * SLAM问题只接受一些特定的Vertex和Edge
     * 如果是通用问题那么hessian是稠密的，除非用户设定某些vertex为marginalized
     */
    enum class ProblemType {
        SLAM_PROBLEM,
        GENERIC_PROBLEM
    };

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    Problem(ProblemType problemType);

    ~Problem();

    bool addVertex(std::shared_ptr<Vertex> vertex);

    /**
     * remove a vertex
     * @param vertex_to_remove
     */
    bool removeVertex(std::shared_ptr<Vertex> vertex);

    bool addEdge(std::shared_ptr<Edge> edge);

    bool removeEdge(std::shared_ptr<Edge> edge);

    /**
     * 取得在优化中被判断为outlier部分的边，方便前端去除outlier
     * @param outlier_edges
     */
    void getOutlierEdges(std::vector<std::shared_ptr<Edge>>& outlier_edges);

    /**
     * 求解此问题
     * @param iterations
     * @return
     */
    bool solve(int iterations = 10);
    bool solveLM(int iterations = 10);
    bool solveDogleg(int iterations = 10);
    bool isGoodStepDogleg();

    /// 边缘化一个frame和以它为host的landmark
    bool marginalize(std::shared_ptr<Vertex> frameVertex,
                     const std::vector<std::shared_ptr<Vertex>>& landmarkVerticies);

    bool marginalize(const std::shared_ptr<Vertex> frameVertex);
    bool marginalize(const std::vector<std::shared_ptr<Vertex> > frameVertex, int pose_dim);

    MatXX getHessianPrior() const { return H_prior_; }
    VecX getbPrior() const { return b_prior_; }
    VecX getErrPrior() const { return err_prior_; }
    MatXX getJtPrior() const { return Jt_prior_inv_; }

    void setHessianPrior(const MatXX& H) { H_prior_ = H; }
    void setbPrior(const VecX& b) { b_prior_ = b; }
    void setErrPrior(const VecX& b) { err_prior_ = b; }
    void setJtPrior(const MatXX& J) { Jt_prior_inv_ = J; }

    void extendHessiansPriorSize(int dim);

    //test compute prior
    void testComputePrior();

public:
    //double total_time = 0.0; // 图像处理时间，正常应放在private中，创建public get接口
    double hessian_time_per_frame = 0.0;
    double time_per_frame = 0.0;
    long solve_count_per_frame = 0;

    // 用于在线程之间共享的数据
    MatXX m_H;
    VecX m_b;
    std::mutex m_mutex; // 一定的加上std::

private:
    /// 三种方式的makeHessian
    void makeHessianNormal();
    void makeHessianOpenMP();
    void makeHessianMultiThread();
    /// 处理边的线程函数，被makeHessianMultiThread()调用
    void thdDoEdges(int start, int end);

    /// Solve的实现，解通用问题
    bool solveGenericProblem(int iterations);

    /// Solve的实现，解SLAM问题
    bool solveSLAMProblem(int iterations);

    /// 设置各顶点的ordering_index
    void setOrdering();

    /// set ordering for new vertex in slam problem
    void addOrderingSLAM(std::shared_ptr<Vertex> v);

    /// 构造大H矩阵
    void makeHessian();

    /// schur求解SBA
    void schurSBA();

    /// 解线性方程
    void solveLinearSystem();

    /// 更新状态变量
    void updateStates();

    void rollbackStates(); // 有时候 update 后残差会变大，需要退回去，重来

    /// 计算并更新Prior部分
    void computePrior();

    /// 判断一个顶点是否为Pose顶点
    bool isPoseVertex(std::shared_ptr<Vertex> v);

    /// 判断一个顶点是否为landmark顶点
    bool isLandmarkVertex(std::shared_ptr<Vertex> v);

    /// 在新增顶点后，需要调整几个hessian的大小
    void resizePoseHessiansWhenAddingPose(std::shared_ptr<Vertex> v);

    /// 检查ordering是否正确
    bool checkOrdering();

    void logoutVectorSize();

    /// 获取某个顶点连接到的边
    std::vector<std::shared_ptr<Edge>> getConnectedEdges(std::shared_ptr<Vertex> vertex);

    /// Levenberg
    /// 计算LM算法的初始Lambda
    void computeLambdaInitLM();

    /// Hessian 对角线加上或者减去  Lambda
    void addLambdatoHessianLM();

    void removeLambdaHessianLM();

    /// LM 算法中用于判断 Lambda 在上次迭代中是否可以，以及Lambda怎么缩放
    bool isGoodStepInLM();

    /// PCG 迭代线性求解器
    VecX pcgSolver(const MatXX &A, const VecX &b, int maxIter);

    double currentLambda_;
    double currentChi_;
    double stopThresholdLM_;    // LM 迭代退出阈值条件
    double ni_;                 //控制 Lambda 缩放大小

    ProblemType problemType_;

    /// 整个信息矩阵
    MatXX Hessian_;
    VecX b_;
    VecX delta_x_;

    /// 先验部分信息
    MatXX H_prior_;
    VecX b_prior_;
    VecX b_prior_backup_;
    VecX err_prior_backup_;

    MatXX Jt_prior_inv_;
    VecX err_prior_;

    /// SBA的Pose部分
    MatXX H_pp_schur_;
    VecX b_pp_schur_;
    // Heesian 的 Landmark 和 pose 部分
    MatXX H_pp_;
    VecX b_pp_;
    MatXX H_ll_;
    VecX b_ll_;

    /// all vertices
    HashVertex verticies_;

    /// all edges
    HashEdge edges_;

    /// 由vertex id查询edge
    HashVertexIdToEdge vertexToEdge_;

    /// Ordering related
    ulong ordering_poses_ = 0;
    ulong ordering_landmarks_ = 0;
    ulong ordering_generic_ = 0;
    std::map<ulong, std::shared_ptr<Vertex>> idx_pose_vertices_;        // 以ordering排序的pose顶点
    std::map<ulong, std::shared_ptr<Vertex>> idx_landmark_vertices_;    // 以ordering排序的landmark顶点

    // verticies need to marg. <Ordering_id_, Vertex>
    HashVertex verticies_marg_;

    bool bDebug = false;
    double t_hessian_cost_ = 0.0;
    double t_PCGsovle_cost_ = 0.0;
};
}
}

#endif //SIMPLE_VIO_PROBLEM_H
