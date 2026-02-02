#include "op.hpp"
#include "cpu/rope_cpu.hpp" // 引入 CPU 头文件

namespace llaisys::ops {

// RoPE: 旋转位置编码
// out: [Seq, Head, Dim]
// in: [Seq, Head, Dim]
// pos_ids: [Seq] (Int64)
// theta: 频率基数 (通常 10000.0)
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    // 1. 检查设备
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        // 2. 检查 pos_ids 类型
        if (pos_ids->dtype() != LLAISYS_DTYPE_I64) {
            throw std::runtime_error("RoPE: pos_ids must be Int64");
        }

        // 3. 获取维度信息
        // 假设形状是 [SeqLen, NHead, HeadDim]
        // 有些实现可能是 [Batch, Seq, Head, Dim]，但作业提示是 [seqlen, nhead, d]
        size_t seq_len = in->shape()[0];
        size_t n_head = in->shape()[1];
        size_t head_dim = in->shape()[2];

        // 4. 调用 CPU 实现
        return cpu::rope(
            out->data(),
            in->data(),
            pos_ids->data(),
            theta,
            seq_len,
            n_head,
            head_dim,
            out->dtype()
        );
    }
    
    throw std::runtime_error("RoPE: device not supported");
}

} // namespace llaisys::ops