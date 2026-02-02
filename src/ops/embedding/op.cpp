#include "op.hpp"
#include "cpu/embedding_cpu.hpp" // 引入 CPU 头文件

namespace llaisys::ops {

// 任务要求：从 weight 中复制 index 指定的行到 out
// out: [num_tokens, hidden_size]
// index: [num_tokens] (必须是 Int64)
// weight: [vocab_size, hidden_size]
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    // 1. 检查设备
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        // 2. 检查 index 类型 (题目要求必须是 Int64)
        if (index->dtype() != LLAISYS_DTYPE_I64) {
            throw std::runtime_error("Embedding: index must be Int64");
        }

        // 3. 获取维度信息
        // index 是 1D 张量，长度就是 token 数量
        size_t num_tokens = index->numel(); 
        // weight 的最后一维是 hidden_size (每个词向量的长度)
        size_t hidden_size = weight->shape().back();
        // weight 的第一维是 vocab_size (词表大小，用于检查越界)
        size_t vocab_size = weight->shape()[0];

        // 4. 调用 CPU 实现
        return cpu::embedding(
            out->data(), 
            index->data(), 
            weight->data(), 
            num_tokens, 
            vocab_size, 
            hidden_size, 
            out->dtype() // 数据的类型 (Float32/Float16/BFloat16)
        );
    }
    
    throw std::runtime_error("Embedding: device not supported");
}

} // namespace llaisys::ops