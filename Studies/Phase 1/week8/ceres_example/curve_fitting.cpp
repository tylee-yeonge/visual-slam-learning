#include <iostream>
#include <ceres/ceres.h>
#include <vector>
#include <cmath>
#include <random>

// 1️⃣ 비용 함수 정의
struct ExponentialResidual {
    ExponentialResidual(double x, double y)
        : x_(x), y_(y) {}
    
    // Ceres가 호출할 함수
    // params[0] = a, params[1] = b
    template <typename T>
    bool operator()(const T* const params, T* residual) const {
        // residual = 측정값 - 예측값
        residual[0] = T(y_) - params[0] * exp(params[1] * T(x_));
        return true;
    }
    
    // Factory method for AutoDiffCostFunction
    static ceres::CostFunction* Create(double x, double y) {
        return new ceres::AutoDiffCostFunction<ExponentialResidual, 1, 2>(
            new ExponentialResidual(x, y));
    }
    
private:
    const double x_;
    const double y_;
};

int main() {
    // 🎲 데이터 생성 (실제 값: a=2.5, b=0.3)
    std::vector<double> x_data, y_data;
    std::default_random_engine generator;
    std::normal_distribution<double> noise(0.0, 0.1);
    
    const double true_a = 2.5;
    const double true_b = 0.3;
    
    for (int i = 0; i < 50; ++i) {
        double x = i * 0.1;
        double y = true_a * exp(true_b * x) + noise(generator);
        x_data.push_back(x);
        y_data.push_back(y);
    }
    
    // 2️⃣ 초기 추정값 (일부러 틀리게)
    double params[2] = {1.0, 0.1};
    
    std::cout << "초기값: a = " << params[0] 
              << ", b = " << params[1] << std::endl;
    
    // 3️⃣ Problem 생성
    ceres::Problem problem;
    
    // 4️⃣ 각 데이터 포인트에 대해 ResidualBlock 추가
    for (size_t i = 0; i < x_data.size(); ++i) {
        ceres::CostFunction* cost_function = 
            ExponentialResidual::Create(x_data[i], y_data[i]);
        
        problem.AddResidualBlock(
            cost_function,      // 비용 함수
            nullptr,            // loss function (nullptr = squared loss)
            params);            // 최적화할 변수
    }
    
    // 5️⃣ Solver 옵션 설정
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 100;
    
    // 6️⃣ 최적화 실행
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    // 7️⃣ 결과 출력
    std::cout << "\n" << summary.BriefReport() << "\n\n";
    std::cout << "최적화 결과:\n";
    std::cout << "  a = " << params[0] << " (실제: " << true_a << ")\n";
    std::cout << "  b = " << params[1] << " (실제: " << true_b << ")\n";
    
    return 0;
}
