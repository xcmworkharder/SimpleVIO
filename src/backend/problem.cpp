#include "backend/problem.h"
#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <iomanip>
#include "utility/tic_toc.h"
#include <omp.h>
#include <thread>
//#ifdef USE_OPENMP
//#include <omp.h>
//#endif
#pragma omp declare reduction (+: VecX: omp_out=omp_out+omp_in)\
     initializer(omp_priv=VecX::Zero(omp_orig.size()))
#pragma omp declare reduction (+: MatXX: omp_out=omp_out+omp_in)\
     initializer(omp_priv=MatXX::Zero(omp_orig.rows(), omp_orig.cols()))

using namespace std;

// define the format you want, you only need one instance of this...
const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, 
                                       Eigen::DontAlignCols, ", ", "\n");

void writeToCSVfile(std::string& name, Eigen::MatrixXd& matrix) {
    std::ofstream f(name.c_str());
    f << matrix.format(CSVFormat);
}

namespace myslam {
namespace backend {

// 控制使用LM或者Dogleg 0: LM, 1: Dogleg
const int algorithm_option = 1;
// 控制是否加速,以及加速的方法 0:不加速 1:openmp 2:multiThread
const int acc_option = 1;
// LM的isGoodStep策略 0:原策略 1:原策略优化 2:新策略 3:新策略2
const int lm_strategy_option = 2;
// DogLeg的isGoodStep策略 0:论文rho_计算策略 1:和g2o一致的策略
const int dogleg_strategy_option = 1;
// 控制LM或者Dogleg算法chi和lambda初始化选项
// 0:Nielsen 1:Levenberg 2:Marquardt 3:Quadratic 4:Dogleg
const int chi_lambda_init_option = 4;

void Problem::logoutVectorSize() {
    // LOG(INFO) <<
    //           "1 problem::LogoutVectorSize verticies_:" << verticies_.size() <<
    //           " edges:" << edges_.size();
}

Problem::Problem(ProblemType problemType) : problemType_(problemType) {
    logoutVectorSize();
    verticies_marg_.clear();
}

Problem::~Problem() {
    // std::cout << "Problem IS Deleted"<<std::endl;
    global_vertex_id = 0;
}

bool Problem::addVertex(std::shared_ptr<Vertex> vertex) {
    if (verticies_.find(vertex->id()) != verticies_.end()) {
        // LOG(WARNING) << "Vertex " << vertex->Id() << " has been added before";
        return false;
    } else {
        verticies_.insert(pair<ulong, shared_ptr<Vertex>>(vertex->id(), vertex));
    }

    if (problemType_ == ProblemType::SLAM_PROBLEM) {
        if (isPoseVertex(vertex)) {
            resizePoseHessiansWhenAddingPose(vertex);
        }
    }
    return true;
}

void Problem::addOrderingSLAM(std::shared_ptr<myslam::backend::Vertex> v) {
    if (isPoseVertex(v)) {
        v->setOrderingId(ordering_poses_);
        idx_pose_vertices_.insert(pair<ulong, std::shared_ptr<Vertex>>(v->id(), v));
        ordering_poses_ += v->localDimension();
    } else if (isLandmarkVertex(v)) {
        v->setOrderingId(ordering_landmarks_);
        ordering_landmarks_ += v->localDimension();
        idx_landmark_vertices_.insert(pair<ulong, std::shared_ptr<Vertex>>(v->id(), v));
    }
}

void Problem::resizePoseHessiansWhenAddingPose(shared_ptr<Vertex> v) {
    int size = H_prior_.rows() + v->localDimension();
    H_prior_.conservativeResize(size, size);
    b_prior_.conservativeResize(size);

    b_prior_.tail(v->localDimension()).setZero();
    H_prior_.rightCols(v->localDimension()).setZero();
    H_prior_.bottomRows(v->localDimension()).setZero();
}

void Problem::extendHessiansPriorSize(int dim) {
    int size = H_prior_.rows() + dim;
    H_prior_.conservativeResize(size, size);
    b_prior_.conservativeResize(size);

    b_prior_.tail(dim).setZero();
    H_prior_.rightCols(dim).setZero();
    H_prior_.bottomRows(dim).setZero();
}

bool Problem::isPoseVertex(std::shared_ptr<myslam::backend::Vertex> v) {
    string type = v->typeInfo();
    return type == string("VertexPose") ||
           type == string("VertexSpeedBias");
}

bool Problem::isLandmarkVertex(std::shared_ptr<myslam::backend::Vertex> v) {
    string type = v->typeInfo();
    return type == string("VertexPointXYZ") ||
           type == string("VertexInverseDepth");
}

bool Problem::addEdge(shared_ptr<Edge> edge) {
    if (edges_.find(edge->id()) == edges_.end()) {
        edges_.insert(pair<ulong, std::shared_ptr<Edge>>(edge->id(), edge));
    } else {
        // LOG(WARNING) << "Edge " << edge->Id() << " has been added before!";
        return false;
    }

    for (auto& vertex: edge->verticies()) {
        vertexToEdge_.insert(pair<ulong, shared_ptr<Edge>>(vertex->id(), edge));
    }
    return true;
}

vector<shared_ptr<Edge>> Problem::getConnectedEdges(std::shared_ptr<Vertex> vertex) {
    vector<shared_ptr<Edge>> edges;
    auto range = vertexToEdge_.equal_range(vertex->id());
    for (auto iter = range.first; iter != range.second; ++iter) {
        // 并且这个edge还需要存在，而不是已经被remove了
        if (edges_.find(iter->second->id()) == edges_.end())
            continue;

        edges.emplace_back(iter->second);
    }
    return edges;
}

bool Problem::removeVertex(std::shared_ptr<Vertex> vertex) {
    //check if the vertex is in map_verticies_
    if (verticies_.find(vertex->id()) == verticies_.end()) {
        // LOG(WARNING) << "The vertex " << vertex->Id() << " is not in the problem!" << endl;
        return false;
    }

    // 这里要 remove 该顶点对应的 edge.
    vector<shared_ptr<Edge>> remove_edges = getConnectedEdges(vertex);
    for (size_t i = 0; i < remove_edges.size(); i++) {
        removeEdge(remove_edges[i]);
    }

    if (isPoseVertex(vertex))
        idx_pose_vertices_.erase(vertex->id());
    else
        idx_landmark_vertices_.erase(vertex->id());

    vertex->setOrderingId(-1);      // used to debug
    verticies_.erase(vertex->id());
    vertexToEdge_.erase(vertex->id());

    return true;
}

bool Problem::removeEdge(std::shared_ptr<Edge> edge) {
    //check if the edge is in map_edges_
    if (edges_.find(edge->id()) == edges_.end()) {
        // LOG(WARNING) << "The edge " << edge->Id() << " is not in the problem!" << endl;
        return false;
    }

    edges_.erase(edge->id());
    return true;
}

bool Problem::solve(int iterations) {
    bool result = false;
    int option = algorithm_option; // 0: LM 1:Dogleg
    switch (option) {
        case 0:
            result = solveLM(iterations);
            break;
        case 1:
            result = solveDogleg(iterations);
            break;
        default:
        std::cerr << "Unknown solve option: " << option << std::endl;
            result = false;
            break;
    }

    return result;
}

bool Problem::solveDogleg(int iterations) {
    if (edges_.size() == 0 || verticies_.size() == 0) {
        std::cerr << "Cannot solve problem without edges or verticies" << std::endl;
        return false;
    }

    TicToc t_solve;
    setOrdering();
    makeHessian();
    computeLambdaInitLM();
    bool found = false;
    radius_ = 1e4;

    int iter = 0;
    const int numIterationsMax = 10;
    double last_chi_ = 1e20;
    while (!found && (iter < numIterationsMax)) {
        std::cout << "iter: " << iter << " , chi = " << currentChi_ << " ,  radius = " << radius_ << std::endl;
        iter++;
        bool oneStepSuccess = false;
        int false_cnt = 0;
        while (!oneStepSuccess && false_cnt < 10) {
            double alpha_ = b_.squaredNorm() / ((Hessian_ * b_).dot(b_));
            h_sd_ = alpha_ * b_;
            h_gn_ = Hessian_.ldlt().solve(b_);
            double h_sd_norm = h_sd_.norm();
            double h_gn_norm = h_gn_.norm();
            if (h_gn_norm <= radius_) {
                h_dl_ = h_gn_;
            } else if (alpha_ * h_sd_norm >= radius_) {
                h_dl_ = (radius_ / h_sd_norm) * h_sd_;
            } else {
                VecX a = h_sd_;
                VecX b = h_gn_;
                double c = a.dot(b - a);
                if (c <= 0) {
                    beta_ = (-c + sqrt(c * c + (b - a).squaredNorm() * (radius_ * radius_ - a.squaredNorm())))
                            / (b -a).squaredNorm();
                } else {
                    beta_ = (radius_ * radius_ - a.squaredNorm()) / (c + sqrt(c * c + (b - a).squaredNorm()
                    * (radius_*radius_ - a.squaredNorm())));
                }
                assert(beta_ > 0.0 && beta_ < 1.0 && "Error while computing beta");
                h_dl_ = a + beta_ * (b - a);
            }
            delta_x_ = h_dl_;

            updateStates();
            oneStepSuccess = isGoodStepInDogleg();
            if (oneStepSuccess) {
                makeHessian();
                false_cnt = 0;
            } else {
                false_cnt++;
                rollbackStates();
            }
        }
        iter++;

        if (last_chi_ - currentChi_ < 1e-5 || b_.norm() < 1e-5) {
            std::cout << "Dogleg: find the right result. " << std::endl;
            found = true;
        }
        last_chi_ = currentChi_;
    }
    std::cout << "problem solve cost: " << t_solve.toc() << " ms" << std::endl;
    std::cout << " makeHessian cost: " << t_hessian_cost_ << " ms" << std::endl;
    hessian_time_per_frame = t_hessian_cost_;
    time_per_frame = t_solve.toc();
    solve_count_per_frame = iter;
    t_hessian_cost_ = 0.0;
    return true;
}

// Dogleg策略因子，用于判断Lambad在上次迭代中是否可以，以及Lambda怎么缩放
bool Problem::isGoodStepInDogleg() {
    double tempChi = 0.0;
    for (auto edge: edges_) {
        edge.second->computeResidual();
        tempChi += edge.second->robustChi2();
    }
    if (err_prior_.size() > 0)
        tempChi += err_prior_.norm();
    tempChi *= 0.5;          // 1/2 * err^2

    // 计算rho
    double rho_ ;
    int option = dogleg_strategy_option;  // 0: 论文策略;  1: 和g2o一致的策略
    switch ( option ) {
        case 0:{  // 论文策略, 计算 rho
            // scale 即为论文中的 L(0) - L(h_dl), 参看 论文 4°和 3.20a 的公式说明
            double scale=0.0;
            if(h_dl_ == h_gn_){
                scale = currentChi_;
            } else if(h_dl_ == radius_ * b_ / b_.norm()) {
                scale = radius_ * (2 * (alpha_ * b_).norm() - radius_) / (2 * alpha_);
            } else {
                scale = 0.5 * alpha_ * pow( (1 - beta_), 2) * b_.squaredNorm()
                        + beta_ * (2 - beta_) * currentChi_;
            }
            // rho = ( F(x) - F(x_new) ) / ( L(0) - L(h_dl) )
            rho_ = ( currentChi_ - tempChi )/ scale;
            break;
        }
        case 1: {  // 按照 g2o 方式 计算 rho_
            double linearGain = - double(delta_x_.transpose() * Hessian_ * delta_x_)
                                + 2 * b_.dot(delta_x_);
            rho_ = ( currentChi_ - tempChi ) / linearGain;
            break;
        }
    }

    // 以下 按照 论文方式更新 radius_
    if (rho_ > 0.75 && isfinite(tempChi)) {
        radius_ = std::max(radius_, 3 * delta_x_.norm());
    }
    else if (rho_ < 0.25) {
        radius_ = std::max(radius_ * 0.5, 1e-7);
    } else {
        // do nothing
    }
    if (rho_ > 0 && isfinite(tempChi)) {
        currentChi_ = tempChi;
        return true;
    } else {
        return false;
    }
}

bool Problem::solveLM(int iterations) {
    if (edges_.size() == 0 || verticies_.size() == 0) {
        std::cerr << "\nCannot solve problem without edges or verticies" << std::endl;
        return false;
    }

    TicToc t_solve;
    // 统计优化变量的维数，为构建 H 矩阵做准备
    setOrdering();
    // 遍历edge, 构建 H 矩阵
    makeHessian();
    // LM 初始化
    computeLambdaInitLM();
    // LM 算法迭代求解
    bool stop = false;
    int iter = 0;
    double last_chi_ = 1e20;
    while (!stop && (iter < iterations)) {
        std::cout << "iter: " << iter << " , chi= " << currentChi_
                  << " , Lambda= " << currentLambda_ << std::endl;
        bool oneStepSuccess = false;
        int false_cnt = 0;
        while (!oneStepSuccess && false_cnt < 10)  {// 不断尝试 Lambda, 直到成功迭代一步
            // setLambda
//            AddLambdatoHessianLM();
            // 第四步，解线性方程
            solveLinearSystem();
            //
//            RemoveLambdaHessianLM();

            // 优化退出条件1： delta_x_ 很小则退出
//            if (delta_x_.squaredNorm() <= 1e-6 || false_cnt > 10)
            // TODO:: 退出条件还是有问题, 好多次误差都没变化了，还在迭代计算，应该搞一个误差不变了就中止
//            if ( false_cnt > 10)
//            {
//                stop = true;
//                break;
//            }

            // 更新状态量
            updateStates();
            // 判断当前步是否可行以及 LM 的 lambda 怎么更新, chi2 也计算一下
            oneStepSuccess = isGoodStepInLM();
            // 后续处理，
            if (oneStepSuccess) {
//                std::cout << " get one step success\n";

                // 在新线性化点 构建 hessian
                makeHessian();
                // TODO:: 这个判断条件可以丢掉，条件 b_max <= 1e-12 很难达到，这里的阈值条件不应该用绝对值，而是相对值
//                double b_max = 0.0;
//                for (int i = 0; i < b_.size(); ++i) {
//                    b_max = max(fabs(b_(i)), b_max);
//                }
//                // 优化退出条件2： 如果残差 b_max 已经很小了，那就退出
//                stop = (b_max <= 1e-12);
                false_cnt = 0;
            } else {
                false_cnt ++;
                rollbackStates();   // 误差没下降，回滚
            }
        }
        iter++;

        // 优化退出条件3： currentChi_ 跟第一次的 chi2 相比，下降了 1e6 倍则退出
        // TODO:: 应该改成前后两次的误差已经不再变化
//        if (sqrt(currentChi_) <= stopThresholdLM_)
//        if (sqrt(currentChi_) < 1e-15)
        if(last_chi_ - currentChi_ < 1e-5) {
            std::cout << "sqrt(currentChi_) <= stopThresholdLM_" << std::endl;
            stop = true;
        }
        last_chi_ = currentChi_;
    }
    std::cout << "problem solve cost: " << t_solve.toc() << " ms" << std::endl;
    std::cout << "   makeHessian cost: " << t_hessian_cost_ << " ms" << std::endl;
    // 记录本次hessian处理时长
    hessian_time_per_frame = t_hessian_cost_;
    // 记录本次frame时长（包括hessian时长）
    time_per_frame = t_solve.toc();
    // 记录本次frame的求解次数
    solve_count_per_frame = iter;
    t_hessian_cost_ = 0.;
    return true;
}

// Chi 和 Lambda 初始化
void Problem::computeChiInitAndLambdaInit() {
    currentChi_ = 0.0;
    for (auto edge: edges_) {
        // 在MakeHessian()中已经计算了edge.second->ComputeResidual()
        currentChi_ += edge.second->robustChi2();
    }
    if (err_prior_.rows() > 0)
        currentChi_ += err_prior_.squaredNorm();
    currentChi_ *= 0.5;

    maxDiagonal_ = 0;
    ulong size = Hessian_.cols();
    assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
    for (ulong i = 0; i < size; ++i) {
        maxDiagonal_ = std::max(fabs(Hessian_(i, i)), maxDiagonal_);
    }
    maxDiagonal_ = std::min(5e10, maxDiagonal_);

    int option = chi_lambda_init_option; // 0:Nielsen; 1:Levenberg;  2:Marquardt;  3:Quadratic;  4:Doglet
    switch (option) {
        case 0: // NIELSEN:
            computeLambdaInitLM_Nielsen();
            break;
        case 1: // LEVENBERG
            computeLambdaInitLM_Levenberg();
            break;
        case 2:  // MARQUARDT
            computeLambdaInitLM_Marquardt();
            break;
        case 3:  // QUADRATIC
            computeLambdaInitLM_Quadratic();
            break;
        case 4:  // DOGLEG
            computeLambdaInitDogleg();
            break;
        default:
            cout << "Please choose correct LM strategy in .ymal file: 0 Nielsen; 1 LevenbergMarquardt; 2 Quadratic" << endl;
            exit(-1);
            break;
    }
}

void Problem::computeLambdaInitLM_Nielsen() {
    ni_ = 2.;
    stopThresholdLM_ = 1e-10 * currentChi_;          // 迭代条件为 误差下降 1e-6 倍
    double tau = 1e-5;  // 1e-5
    currentLambda_ = tau * maxDiagonal_;
    std::cout << "currentLamba_: "<<currentLambda_<<", maxDiagonal: "<<maxDiagonal_<<std::endl;
}

void Problem::computeLambdaInitLM_Levenberg() {
    currentLambda_ = 1e-2;
    lastLambda_ = currentLambda_;
}

void Problem::computeLambdaInitLM_Marquardt() {
    double tau = 1e-5;  // 1e-5
    currentLambda_ = tau * maxDiagonal_;
}

void Problem::computeLambdaInitLM_Quadratic() {
    double tau = 1e-5;  // 1e-5
    currentLambda_ = tau * maxDiagonal_;
}

void Problem::computeLambdaInitDogleg() {
    currentLambda_ = 1e-7;
}

bool Problem::solveGenericProblem(int iterations) {
    return true;
}

void Problem::setOrdering() {
    // 每次重新计数
    ordering_poses_ = 0;
    ordering_generic_ = 0;
    ordering_landmarks_ = 0;

    // Note:: verticies_ 是 map 类型的, 顺序是按照 id 号排序的
    for (auto vertex : verticies_) {
        ordering_generic_ += vertex.second->localDimension();  // 所有的优化变量总维数
        if (problemType_ == ProblemType::SLAM_PROBLEM) {  // 如果是 slam 问题，还要分别统计 pose 和 landmark 的维数，后面会对他们进行排序
            addOrderingSLAM(vertex.second);
        }

    }

    if (problemType_ == ProblemType::SLAM_PROBLEM) {
        // 这里要把 landmark 的 ordering 加上 pose 的数量，就保持了 landmark 在后,而 pose 在前
        ulong all_pose_dimension = ordering_poses_;
        for (auto landmarkVertex : idx_landmark_vertices_) {
            landmarkVertex.second->setOrderingId(
                    landmarkVertex.second->orderingId() + all_pose_dimension);
        }
    }
//    CHECK_EQ(CheckOrdering(), true);
}

bool Problem::checkOrdering() {
    if (problemType_ == ProblemType::SLAM_PROBLEM) {
        int current_ordering = 0;
        for (auto v: idx_pose_vertices_) {
            assert(v.second->orderingId() == current_ordering);
            current_ordering += v.second->localDimension();
        }
        for (auto v: idx_landmark_vertices_) {
            assert(v.second->orderingId() == current_ordering);
            current_ordering += v.second->localDimension();
        }
    }
    return true;
}

/// 支持三种方式：不加速 openmp加速 和 多线程加速
void Problem::makeHessian() {
    int option = 1; // 0: normal 1:openmp 2:multithread
    switch (option) {
        case 0:
            makeHessianNormal();
            break;
        case 1:
            makeHessianOpenMP();
            break;
        case 2:
            makeHessianMultiThread();
            break;
    }
}

void Problem::makeHessianNormal() {
    TicToc t_h;
    // 直接构造大的 H 矩阵
    ulong size = ordering_generic_;
    MatXX H(MatXX::Zero(size, size));
    VecX b(VecX::Zero(size));

    for (auto& edge : edges_) {
        edge.second->computeResidual();
        edge.second->computeJacobians();

        // TODO:: robust cost
        auto jacobians = edge.second->jacobians();
        auto verticies = edge.second->verticies();
        assert(jacobians.size() == verticies.size());
        for (size_t i = 0; i < verticies.size(); ++i) {
            auto v_i = verticies[i];
            if (v_i->isFixed()) continue;    // Hessian 里不需要添加它的信息，也就是它的雅克比为 0

            auto jacobian_i = jacobians[i];
            ulong index_i = v_i->orderingId();
            ulong dim_i = v_i->localDimension();

            // 鲁棒核函数会修改残差和信息矩阵，如果没有设置 robust cost function，就会返回原来的
            double drho;
            MatXX robustInfo(edge.second->information().rows(),edge.second->information().cols());
            edge.second->robustInfo(drho,robustInfo);

            MatXX JtW = jacobian_i.transpose() * robustInfo;
            for (size_t j = i; j < verticies.size(); ++j) {
                auto v_j = verticies[j];

                if (v_j->isFixed()) continue;

                auto jacobian_j = jacobians[j];
                ulong index_j = v_j->orderingId();
                ulong dim_j = v_j->localDimension();

                assert(v_j->orderingId() != -1);
                MatXX hessian = JtW * jacobian_j;

                // 所有的信息矩阵叠加起来
                H.block(index_i, index_j, dim_i, dim_j).noalias() += hessian;
                if (j != i) {
                    // 对称的下三角
                    H.block(index_j, index_i, dim_j, dim_i).noalias() += hessian.transpose();

                }
            }
            b.segment(index_i, dim_i).noalias() -=
                    drho * jacobian_i.transpose()* edge.second->information()
                    * edge.second->residual();
        }

    }
    Hessian_ = H;
    b_ = b;
    t_hessian_cost_ += t_h.toc();

    if (H_prior_.rows() > 0) {
        MatXX H_prior_tmp = H_prior_;
        VecX b_prior_tmp = b_prior_;

        /// 遍历所有 POSE 顶点，然后设置相应的先验维度为 0 .  fix 外参数, SET PRIOR TO ZERO
        /// landmark 没有先验
        for (auto vertex : verticies_) {
            if (isPoseVertex(vertex.second) && vertex.second->isFixed() ) {
                int idx = vertex.second->orderingId();
                int dim = vertex.second->localDimension();
                H_prior_tmp.block(idx,0, dim, H_prior_tmp.cols()).setZero();
                H_prior_tmp.block(0,idx, H_prior_tmp.rows(), dim).setZero();
                b_prior_tmp.segment(idx,dim).setZero();
//                std::cout << " fixed prior, set the Hprior and bprior part to zero, idx: "<<idx <<" dim: "<<dim<<std::endl;
            }
        }
        Hessian_.topLeftCorner(ordering_poses_, ordering_poses_) += H_prior_tmp;
        b_.head(ordering_poses_) += b_prior_tmp;
    }
    delta_x_ = VecX::Zero(size);  // initial delta_x = 0_n;
}

void Problem::makeHessianOpenMP() {
    TicToc t_h;
    // 直接构造大的 H 矩阵
    ulong size = ordering_generic_;
    MatXX H(MatXX::Zero(size, size));
    VecX b(VecX::Zero(size));

    // 使用openmp加速
    vector<unsigned long> edge_ids;
    for (auto& edge : edges_) {
        edge_ids.push_back(edge.first);
    }
    int threadNum = 6;
    omp_set_num_threads(threadNum);
    Eigen::setNbThreads(1);
#pragma omp parallel for reduction(+:H) reduction(+:b)
    for (int idx = 0; idx < edges_.size(); ++idx) {
        auto edge = edges_[edge_ids[idx]];
        edge->computeResidual();
        edge->computeJacobians();

        // TODO:: robust cost
        auto jacobians = edge->jacobians();
        auto verticies = edge->verticies();
        assert(jacobians.size() == verticies.size());
        for (size_t i = 0; i < verticies.size(); ++i) {
            auto v_i = verticies[i];
            if (v_i->isFixed()) continue;    // Hessian 里不需要添加它的信息，也就是它的雅克比为 0

            auto jacobian_i = jacobians[i];
            ulong index_i = v_i->orderingId();
            ulong dim_i = v_i->localDimension();

            // 鲁棒核函数会修改残差和信息矩阵，如果没有设置 robust cost function，就会返回原来的
            double drho;
            MatXX robustInfo(edge->information().rows(),edge->information().cols());
            edge->robustInfo(drho,robustInfo);

            MatXX JtW = jacobian_i.transpose() * robustInfo;
            for (size_t j = i; j < verticies.size(); ++j) {
                auto v_j = verticies[j];

                if (v_j->isFixed()) continue;

                auto jacobian_j = jacobians[j];
                ulong index_j = v_j->orderingId();
                ulong dim_j = v_j->localDimension();

                assert(v_j->orderingId() != -1);
                MatXX hessian = JtW * jacobian_j;

                // 所有的信息矩阵叠加起来
                H.block(index_i, index_j, dim_i, dim_j).noalias() += hessian;
                if (j != i) {
                    // 对称的下三角
                    H.block(index_j, index_i, dim_j, dim_i).noalias() += hessian.transpose();

                }
            }
            b.segment(index_i, dim_i).noalias() -=
                    drho * jacobian_i.transpose()* edge->information() * edge->residual();
        }

    }
    Hessian_ = H;
    b_ = b;
    t_hessian_cost_ += t_h.toc();

    if (H_prior_.rows() > 0) {
        MatXX H_prior_tmp = H_prior_;
        VecX b_prior_tmp = b_prior_;

        /// 遍历所有 POSE 顶点，然后设置相应的先验维度为 0 .  fix 外参数, SET PRIOR TO ZERO
        /// landmark 没有先验
        for (auto vertex : verticies_) {
            if (isPoseVertex(vertex.second) && vertex.second->isFixed() ) {
                int idx = vertex.second->orderingId();
                int dim = vertex.second->localDimension();
                H_prior_tmp.block(idx,0, dim, H_prior_tmp.cols()).setZero();
                H_prior_tmp.block(0,idx, H_prior_tmp.rows(), dim).setZero();
                b_prior_tmp.segment(idx,dim).setZero();
//                std::cout << " fixed prior, set the Hprior and bprior part to zero, idx: "<<idx <<" dim: "<<dim<<std::endl;
            }
        }
        Hessian_.topLeftCorner(ordering_poses_, ordering_poses_) += H_prior_tmp;
        b_.head(ordering_poses_) += b_prior_tmp;
    }
    delta_x_ = VecX::Zero(size);  // initial delta_x = 0_n;

    Eigen::setNbThreads(threadNum);
}

void Problem::makeHessianMultiThread() {
    TicToc t_h;
    ulong size = ordering_generic_;
    m_H.setZero(size, size);
    m_b.setZero(size);

    int thd_num = 6;
    thread thd[thd_num];
    int start = 0, end = 0;
    cout << "Total edges: " << edges_.size() << endl;
    for (int i = 1; i <= thd_num; ++i) {
        end = edges_.size() * i / thd_num;
        thd[i - 1] = thread(mem_fn(&Problem::thdDoEdges), this, start, end - 1);
        thd[i - 1].join();
        start = end;
    }

//    for (int i = 0; i < thd_num; ++i) {
//        thd[i].join();
//    }

    Hessian_ = m_H;
    b_ = m_b;
    t_hessian_cost_ += t_h.toc();

    if (H_prior_.rows() > 0) {
        MatXX H_prior_tmp = H_prior_;
        VecX b_prior_tmp = b_prior_;

        for (auto vertex : verticies_) {
            if (isPoseVertex(vertex.second) && vertex.second->isFixed()) {
                int idx = vertex.second->orderingId();
                int dim = vertex.second->localDimension();
                H_prior_tmp.block(idx, 0, dim, H_prior_tmp.cols()).setZero();
                H_prior_tmp.block(0, idx, H_prior_tmp.rows(), dim).setZero();
                b_prior_tmp.segment(idx, dim).setZero();
            }
        }
        Hessian_.topLeftCorner(ordering_poses_, ordering_poses_) += H_prior_tmp;
        b_.head(ordering_poses_) += b_prior_tmp;
    }
    delta_x_ = VecX::Zero(size);
}

void Problem::thdDoEdges(int start, int end) {
    auto it = edges_.begin();
    for (int i = 0; i <= end; ++i) {
        if (i < start) {
            ++it;
            continue;
        }

        auto edge = *it;
        edge.second->computeResidual();
        edge.second->computeJacobians();
        auto jacobians = edge.second->jacobians();
        auto verticies = edge.second->verticies();
        assert(jacobians.size() == verticies.size());

        for (size_t i = 0; i < verticies.size(); ++i) {
            auto v_i = verticies[i];
            if (v_i->isFixed()) {
                continue;
            }
            auto jacobian_i = jacobians[i];
            ulong index_i = v_i->orderingId();
            ulong dim_i = v_i->localDimension();

            double drho;
            MatXX robustInfo(edge.second->information().rows(),
                             edge.second->information().cols());
            edge.second->robustInfo(drho, robustInfo);

            MatXX JtW = jacobian_i.transpose() * robustInfo;
            for (size_t j = i; j < verticies.size(); ++j) {
                auto v_j = verticies[j];
                if (v_j->isFixed()) {
                    continue;
                }
                auto jacobian_j = jacobians[j];
                ulong index_j = v_j->orderingId();
                ulong dim_j = v_j->localDimension();
                assert(v_j->orderingId() != -1);
                MatXX hessian = JtW * jacobian_j;
                m_mutex.lock();
                m_H.block(index_i, index_j, dim_i, dim_j).noalias() += hessian;
                if (j != i) {
                    m_H.block(index_j, index_j, dim_j,
                              dim_i).noalias() += hessian.transpose();
                }
                m_mutex.unlock();
            }
            m_mutex.lock();
            m_b.segment(index_i, dim_i).noalias() -=
                    drho * jacobian_i.transpose()
                    * edge.second->information() * edge.second->residual();
            m_mutex.unlock();
        }
    }
}

/*
* Solve Hx = b, we can use PCG iterative method or use sparse Cholesky
*/
void Problem::solveLinearSystem() {
    if (problemType_ == ProblemType::GENERIC_PROBLEM) {
        // PCG solver
        MatXX H = Hessian_;
        for (size_t i = 0; i < Hessian_.cols(); ++i) {
            H(i, i) += currentLambda_;
        }
        // delta_x_ = PCGSolver(H, b_, H.rows() * 2);
        delta_x_ = H.ldlt().solve(b_);

    } else {

//        TicToc t_Hmminv;
        // step1: schur marginalization --> Hpp, bpp
        int reserve_size = ordering_poses_;
        int marg_size = ordering_landmarks_;
        MatXX Hmm = Hessian_.block(reserve_size, reserve_size, marg_size, marg_size);
        MatXX Hpm = Hessian_.block(0, reserve_size, reserve_size, marg_size);
        MatXX Hmp = Hessian_.block(reserve_size, 0, marg_size, reserve_size);
        VecX bpp = b_.segment(0, reserve_size);
        VecX bmm = b_.segment(reserve_size, marg_size);

        // Hmm 是对角线矩阵，它的求逆可以直接为对角线块分别求逆，如果是逆深度，对角线块为1维的，则直接为对角线的倒数，这里可以加速
        MatXX Hmm_inv(MatXX::Zero(marg_size, marg_size));
        // TODO:: use openMP
        for (auto landmarkVertex : idx_landmark_vertices_) {
            int idx = landmarkVertex.second->orderingId() - reserve_size;
            int size = landmarkVertex.second->localDimension();
            Hmm_inv.block(idx, idx, size, size) = Hmm.block(idx, idx, size, size).inverse();
        }

        MatXX tempH = Hpm * Hmm_inv;
        H_pp_schur_ = Hessian_.block(0, 0, ordering_poses_, ordering_poses_) - tempH * Hmp;
        b_pp_schur_ = bpp - tempH * bmm;

        // step2: solve Hpp * delta_x = bpp
        VecX delta_x_pp(VecX::Zero(reserve_size));

        for (ulong i = 0; i < ordering_poses_; ++i) {
            H_pp_schur_(i, i) += currentLambda_;              // LM Method
        }

        // TicToc t_linearsolver;
        delta_x_pp =  H_pp_schur_.ldlt().solve(b_pp_schur_);//  SVec.asDiagonal() * svd.matrixV() * Ub;
        delta_x_.head(reserve_size) = delta_x_pp;
        // std::cout << " Linear Solver Time Cost: " << t_linearsolver.toc() << std::endl;

        // step3: solve Hmm * delta_x = bmm - Hmp * delta_x_pp;
        VecX delta_x_ll(marg_size);
        delta_x_ll = Hmm_inv * (bmm - Hmp * delta_x_pp);
        delta_x_.tail(marg_size) = delta_x_ll;
//        std::cout << "schur time cost: "<< t_Hmminv.toc()<<std::endl;
    }

}

void Problem::updateStates() {
    // update vertex
    for (auto vertex: verticies_) {
        vertex.second->backUpParameters();    // 保存上次的估计值

        ulong idx = vertex.second->orderingId();
        ulong dim = vertex.second->localDimension();
        VecX delta = delta_x_.segment(idx, dim);
        vertex.second->plus(delta);
    }

    // update prior
    if (err_prior_.rows() > 0) {
        // BACK UP b_prior_
        b_prior_backup_ = b_prior_;
        err_prior_backup_ = err_prior_;

        /// update with first order Taylor, b' = b + \frac{\delta b}{\delta x} * \delta x
        /// \delta x = Computes the linearized deviation from the references (linearization points)
        b_prior_ -= H_prior_ * delta_x_.head(ordering_poses_);       // update the error_prior
        err_prior_ = -Jt_prior_inv_ * b_prior_.head(ordering_poses_ - 15);

//        std::cout << "                : "<< b_prior_.norm()<<" " <<err_prior_.norm()<< std::endl;
//        std::cout << "     delta_x_ ex: "<< delta_x_.head(6).norm() << std::endl;
    }
}

void Problem::rollbackStates() {
    // update vertex
    for (auto vertex: verticies_) {
        vertex.second->rollBackParameters();
    }

    // Roll back prior_
    if (err_prior_.rows() > 0) {
        b_prior_ = b_prior_backup_;
        err_prior_ = err_prior_backup_;
    }
}

/// LM
void Problem::computeLambdaInitLM() {
    ni_ = 2.;
    currentLambda_ = -1.;
    currentChi_ = 0.0;

    for (auto edge: edges_) {
        currentChi_ += edge.second->robustChi2();
    }
    if (err_prior_.rows() > 0)
        currentChi_ += err_prior_.norm();
    currentChi_ *= 0.5;

    stopThresholdLM_ = 1e-10 * currentChi_;          // 迭代条件为 误差下降 1e-6 倍

    double maxDiagonal = 0;
    ulong size = Hessian_.cols();
    assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
    for (ulong i = 0; i < size; ++i) {
        maxDiagonal = std::max(fabs(Hessian_(i, i)), maxDiagonal);
    }

    maxDiagonal = std::min(5e10, maxDiagonal);
    double tau = 1e-5;  // 1e-5
    currentLambda_ = tau * maxDiagonal;
//        std::cout << "currentLamba_: "<<maxDiagonal<<" "<<currentLambda_<<std::endl;
}

void Problem::addLambdatoHessianLM() {
    ulong size = Hessian_.cols();
    assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
    for (ulong i = 0; i < size; ++i) {
        Hessian_(i, i) += currentLambda_;
    }
}

void Problem::removeLambdaHessianLM() {
    ulong size = Hessian_.cols();
    assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
    // TODO:: 这里不应该减去一个，数值的反复加减容易造成数值精度出问题？而应该保存叠加lambda前的值，在这里直接赋值
    for (ulong i = 0; i < size; ++i) {
        Hessian_(i, i) -= currentLambda_;
    }
}

bool Problem::isGoodStepInLM() {

    int option = 1; // 1: normal 2:stratety referenced by "The Levenberg-Marguardt..." 3:another
    double scale = 0;
    double tempChi = 0.0;
    double rho = 0.0;
    double h = 0;
    double diff = 0;
    double alpha_ = 0;
    switch (option) {
        case 1:
            scale = 0;
//    scale = 0.5 * delta_x_.transpose() * (currentLambda_ * delta_x_ + b_);
//    scale += 1e-3;    // make sure it's non-zero :)
            scale = 0.5 * delta_x_.transpose() * (currentLambda_ * delta_x_ + b_);
            scale += 1e-6;    // make sure it's non-zero :)

            // recompute residuals after update state
            tempChi = 0.0;
            for (auto edge : edges_) {
                edge.second->computeResidual();
                tempChi += edge.second->robustChi2();
            }
            if (err_prior_.size() > 0)
                tempChi += err_prior_.norm();
            tempChi *= 0.5;          // 1/2 * err^2

            rho = (currentChi_ - tempChi) / scale;
            if (rho > 0 && isfinite(tempChi))  { // last step was good, 误差在下降

                double alpha = 1. - pow((2 * rho - 1), 3);
                alpha = std::min(alpha, 2. / 3.);
                double scaleFactor = (std::max)(1. / 3., alpha);
                currentLambda_ *= scaleFactor;
                ni_ = 2;
                currentChi_ = tempChi;
                return true;
            } else {
                currentLambda_ *= ni_;
                ni_ *= 2;
                return false;
            }
            break;

        case 2:
            rho = (currentChi_ - tempChi) / scale;
            h = delta_x_.transpose() * b_;
            diff = currentChi_ - tempChi;
            alpha_ = h / (0.5 * diff + h);
            if (rho > 0 && isfinite(tempChi)) {
                currentLambda_ = std::max(currentLambda_ / (1 + alpha_), 1e-7);
                currentChi_ = tempChi;
                return true;
            } else if (rho <= 0 && isfinite(tempChi)) {
                currentLambda_ = currentLambda_ + std::abs(diff * 0.5 / alpha_);
                currentChi_ = tempChi;
                return true;
            } else {
                return false;
            }
            break;

        case 3:
            rho = (currentChi_ - tempChi) / scale;
            if (rho < 0.25 && isfinite(tempChi)) {
                currentLambda_ *= 2.0;
                currentChi_ = tempChi;
                return true;
            } else if (rho > 0.75 && isfinite(tempChi)) {
                currentLambda_ /= 3.0;
                currentChi_  = tempChi;
                return true;
            } else {
                return false;
            }
            break;
    }
}

/** @brief conjugate gradient with perconditioning
*
*  the jacobi PCG method
*
*/
VecX Problem::pcgSolver(const MatXX& A, const VecX& b, int maxIter = -1) {
    assert(A.rows() == A.cols() && "PCG solver ERROR: A is not a square matrix");
    int rows = b.rows();
    int n = maxIter < 0 ? rows : maxIter;
    VecX x(VecX::Zero(rows));
    MatXX M_inv = A.diagonal().asDiagonal().inverse();
    VecX r0(b);  // initial r = b - A*0 = b
    VecX z0 = M_inv * r0;
    VecX p(z0);
    VecX w = A * p;
    double r0z0 = r0.dot(z0);
    double alpha = r0z0 / p.dot(w);
    VecX r1 = r0 - alpha * w;
    int i = 0;
    double threshold = 1e-6 * r0.norm();
    while (r1.norm() > threshold && i < n) {
        i++;
        VecX z1 = M_inv * r1;
        double r1z1 = r1.dot(z1);
        double belta = r1z1 / r0z0;
        z0 = z1;
        r0z0 = r1z1;
        r0 = r1;
        p = belta * p + z1;
        w = A * p;
        alpha = r1z1 / p.dot(w);
        x += alpha * p;
        r1 -= alpha * w;
    }
    return x;
}

/*
*  marg 所有和 frame 相连的 edge: imu factor, projection factor
*  如果某个landmark和该frame相连，但是又不想加入marg, 那就把该edge先去掉
*
*/
bool Problem::marginalize(const std::vector<std::shared_ptr<Vertex>> margVertexs,
                          int pose_dim) {
    setOrdering();
    /// 找到需要 marg 的 edge, margVertexs[0] is frame, its edge contained pre-intergration
    std::vector<shared_ptr<Edge>> marg_edges = getConnectedEdges(margVertexs[0]);
    std::unordered_map<int, shared_ptr<Vertex>> margLandmark;
    // 构建 Hessian 的时候 pose 的顺序不变，landmark的顺序要重新设定
    int marg_landmark_size = 0;
//    std::cout << "\n marg edge 1st id: "<< marg_edges.front()->Id() << " end id: "<<marg_edges.back()->Id()<<std::endl;
    for (size_t i = 0; i < marg_edges.size(); ++i) {
//        std::cout << "marg edge id: "<< marg_edges[i]->Id() <<std::endl;
        auto verticies = marg_edges[i]->verticies();
        for (auto iter : verticies) {
            if (isLandmarkVertex(iter) && margLandmark.find(iter->id()) == margLandmark.end()) {
                iter->setOrderingId(pose_dim + marg_landmark_size);
                margLandmark.insert(make_pair(iter->id(), iter));
                marg_landmark_size += iter->localDimension();
            }
        }
    }
//    std::cout << "pose dim: " << pose_dim <<std::endl;
    int cols = pose_dim + marg_landmark_size;
    /// 构建误差 H 矩阵 H = H_marg + H_pp_prior
    MatXX H_marg(MatXX::Zero(cols, cols));
    VecX b_marg(VecX::Zero(cols));
    int ii = 0;
    for (auto edge : marg_edges) {
        edge->computeResidual();
        edge->computeJacobians();
        auto jacobians = edge->jacobians();
        auto verticies = edge->verticies();
        ii++;

        assert(jacobians.size() == verticies.size());
        for (size_t i = 0; i < verticies.size(); ++i) {
            auto v_i = verticies[i];
            auto jacobian_i = jacobians[i];
            ulong index_i = v_i->orderingId();
            ulong dim_i = v_i->localDimension();

            double drho;
            MatXX robustInfo(edge->information().rows(), edge->information().cols());
            edge->robustInfo(drho, robustInfo);

            for (size_t j = i; j < verticies.size(); ++j) {
                auto v_j = verticies[j];
                auto jacobian_j = jacobians[j];
                ulong index_j = v_j->orderingId();
                ulong dim_j = v_j->localDimension();

                MatXX hessian = jacobian_i.transpose() * robustInfo * jacobian_j;

                assert(hessian.rows() == v_i->localDimension() && hessian.cols() == v_j->localDimension());
                // 所有的信息矩阵叠加起来
                H_marg.block(index_i, index_j, dim_i, dim_j) += hessian;
                if (j != i) {
                    // 对称的下三角
                    H_marg.block(index_j, index_i, dim_j, dim_i) += hessian.transpose();
                }
            }
            b_marg.segment(index_i, dim_i) -=
                    drho * jacobian_i.transpose() * edge->information() * edge->residual();
        }

    }
    std::cout << "edge factor cnt: " << ii <<std::endl;

    /// marg landmark
    int reserve_size = pose_dim;
    if (marg_landmark_size > 0) {
        int marg_size = marg_landmark_size;
        MatXX Hmm = H_marg.block(reserve_size, reserve_size, marg_size, marg_size);
        MatXX Hpm = H_marg.block(0, reserve_size, reserve_size, marg_size);
        MatXX Hmp = H_marg.block(reserve_size, 0, marg_size, reserve_size);
        VecX bpp = b_marg.segment(0, reserve_size);
        VecX bmm = b_marg.segment(reserve_size, marg_size);

        // Hmm 是对角线矩阵，它的求逆可以直接为对角线块分别求逆，如果是逆深度，对角线块为1维的，则直接为对角线的倒数，这里可以加速
        MatXX Hmm_inv(MatXX::Zero(marg_size, marg_size));
        // TODO:: use openMP
        for (auto iter: margLandmark) {
            int idx = iter.second->orderingId() - reserve_size;
            int size = iter.second->localDimension();
            Hmm_inv.block(idx, idx, size, size) = Hmm.block(idx, idx, size, size).inverse();
        }

        MatXX tempH = Hpm * Hmm_inv;
        MatXX Hpp = H_marg.block(0, 0, reserve_size, reserve_size) - tempH * Hmp;
        bpp = bpp - tempH * bmm;
        H_marg = Hpp;
        b_marg = bpp;
    }

    VecX b_prior_before = b_prior_;
    if (H_prior_.rows() > 0) {
        H_marg += H_prior_;
        b_marg += b_prior_;
    }

    /// marg frame and speedbias
    int marg_dim = 0;

    // index 大的先移动
    for (int k = margVertexs.size() -1; k >= 0; --k) {

        int idx = margVertexs[k]->orderingId();
        int dim = margVertexs[k]->localDimension();
//        std::cout << k << " "<<idx << std::endl;
        marg_dim += dim;
        // move the marg pose to the Hmm bottown right
        // 将 row i 移动矩阵最下面
        Eigen::MatrixXd temp_rows = H_marg.block(idx, 0, dim, reserve_size);
        Eigen::MatrixXd temp_botRows = H_marg.block(idx + dim, 0, reserve_size - idx - dim, reserve_size);
        H_marg.block(idx, 0, reserve_size - idx - dim, reserve_size) = temp_botRows;
        H_marg.block(reserve_size - dim, 0, dim, reserve_size) = temp_rows;

        // 将 col i 移动矩阵最右边
        Eigen::MatrixXd temp_cols = H_marg.block(0, idx, reserve_size, dim);
        Eigen::MatrixXd temp_rightCols = H_marg.block(0, idx + dim, reserve_size, reserve_size - idx - dim);
        H_marg.block(0, idx, reserve_size, reserve_size - idx - dim) = temp_rightCols;
        H_marg.block(0, reserve_size - dim, reserve_size, dim) = temp_cols;

        Eigen::VectorXd temp_b = b_marg.segment(idx, dim);
        Eigen::VectorXd temp_btail = b_marg.segment(idx + dim, reserve_size - idx - dim);
        b_marg.segment(idx, reserve_size - idx - dim) = temp_btail;
        b_marg.segment(reserve_size - dim, dim) = temp_b;
    }

    double eps = 1e-8;
    int m2 = marg_dim;
    int n2 = reserve_size - marg_dim;   // marg pose
    Eigen::MatrixXd Amm = 0.5 * (H_marg.block(n2, n2, m2, m2)
                                 + H_marg.block(n2, n2, m2, m2).transpose());

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);
    Eigen::MatrixXd Amm_inv = saes.eigenvectors() * Eigen::VectorXd(
            (saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal() *
                              saes.eigenvectors().transpose();

    Eigen::VectorXd bmm2 = b_marg.segment(n2, m2);
    Eigen::MatrixXd Arm = H_marg.block(0, n2, n2, m2);
    Eigen::MatrixXd Amr = H_marg.block(n2, 0, m2, n2);
    Eigen::MatrixXd Arr = H_marg.block(0, 0, n2, n2);
    Eigen::VectorXd brr = b_marg.segment(0, n2);
    Eigen::MatrixXd tempB = Arm * Amm_inv;
    H_prior_ = Arr - tempB * Amr;
    b_prior_ = brr - tempB * bmm2;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(H_prior_);
    Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array(), 0));
    Eigen::VectorXd S_inv = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array().inverse(), 0));

    Eigen::VectorXd S_sqrt = S.cwiseSqrt();
    Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();
    Jt_prior_inv_ = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
    err_prior_ = -Jt_prior_inv_ * b_prior_;

    MatXX J = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
    H_prior_ = J.transpose() * J;
    MatXX tmp_h = MatXX( (H_prior_.array().abs() > 1e-9).select(H_prior_.array(),0) );
    H_prior_ = tmp_h;

    // std::cout << "my marg b prior: " <<b_prior_.rows()<<" norm: "<< b_prior_.norm() << std::endl;
    // std::cout << "    error prior: " <<err_prior_.norm() << std::endl;

    // remove vertex and remove edge
    for (size_t k = 0; k < margVertexs.size(); ++k) {
        removeVertex(margVertexs[k]);
    }

    for (auto landmarkVertex : margLandmark) {
        removeVertex(landmarkVertex.second);
    }
    return true;
}
}
}

