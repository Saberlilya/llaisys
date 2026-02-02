#include "op.hpp"
#include "cpu/linear_cpu.hpp" // 引入 CPU 头文件

namespace llaisys::ops {

// 线性层: Y = X * W^T + b
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    // 1. 检查设备
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        // 2. 获取维度
        // in: [M, K]
        size_t M = in->shape()[0];
        size_t K = in->shape()[1];
        // out: [M, N]
        // weight: [N, K] -> weight->shape()[0] 也是 N
        size_t N = out->shape()[1]; 

        // 3. 调用 CPU 实现
        return cpu::linear(
            out->data(), 
            in->data(), 
            weight->data(), 
            bias ? bias->data() : nullptr, // bias 可选，如果为空传 nullptr
            M, K, N, 
            out->dtype() // 数据类型
        );
    }
    
    throw std::runtime_error("Linear: device not supported");
}

} // namespace llaisys::ops