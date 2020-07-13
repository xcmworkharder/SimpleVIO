#ifndef SIMPLE_VIO_TIC_TOC_H
#define SIMPLE_VIO_TIC_TOC_H

#include <ctime>
#include <cstdlib>
#include <chrono>

/**
 * 用于计算消耗时间
 */
class TicToc {
public:
    TicToc() {
        tic();
    }

    // 获取启动时间
    void tic() {
        start = std::chrono::system_clock::now();
    }

    // 获取结束时间,并计算消耗时间
    double toc() {
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elepsed_seconds = end - start;
        return elepsed_seconds.count() * 1000;
    }

private:
    std::chrono::time_point<std::chrono::system_clock> start, end; // 启动和结束时间
};
#endif //SIMPLE_VIO_TIC_TOC_H
