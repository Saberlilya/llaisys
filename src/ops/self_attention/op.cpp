#include "op.hpp"
#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        // 1. 获取维度
        // q: [seqlen, nhead, d]
        size_t seqlen = q->shape()[0];
        size_t nhead = q->shape()[1];
        size_t d = q->shape()[2];
        
        // k: [total_len, nkvhead, d]
        size_t total_len = k->shape()[0];
        size_t nkvhead = k->shape()[1];
        
        // v: [total_len, nkvhead, dv]
        // 注意：v 的最后一维 dv 可能跟 d 不同 (虽然通常相同)
        size_t dv = v->shape()[2];

        // 2. 调用 CPU 实现
        return cpu::self_attention(
            attn_val->data(),
            q->data(),
            k->data(),
            v->data(),
            seqlen, total_len, nhead, nkvhead, d, dv,
            scale,
            attn_val->dtype()
        );
    }
    
    throw std::runtime_error("SelfAttention: device not supported");
}

} // namespace llaisys::ops