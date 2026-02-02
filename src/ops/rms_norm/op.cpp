#include "op.hpp"
#include "cpu/rms_norm_cpu.hpp" // 引入 CPU 头文件

namespace llaisys::ops {

// RMSNorm: Y = X * W / sqrt(mean(X^2) + eps)
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    // 1. 检查设备
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        // 2. 获取维度
        // 题目假设输入是 2D 连续张量 [Rows, HiddenSize]
        // 实际上哪怕是 3D [Batch, Seq, Hidden]，我们在内存上也可以看作 [Batch*Seq, Hidden]
        size_t hidden_size = in->shape().back(); // 最后一维是特征维度
        size_t num_rows = in->numel() / hidden_size; // 剩下的都是行数

        // 3. 调用 CPU 实现
        return cpu::rms_norm(
            out->data(),
            in->data(),
            weight->data(),
            eps,
            num_rows,
            hidden_size,
            out->dtype()
        );
    }
    
    throw std::runtime_error("RMSNorm: device not supported");
}

} // namespace llaisys::ops