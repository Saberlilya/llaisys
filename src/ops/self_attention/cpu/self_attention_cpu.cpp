#include "self_attention_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <type_traits>

namespace llaisys::ops::cpu {

// --- 类型转换辅助 ---
template <typename T>
inline float val_to_float(T v) {
    if constexpr (std::is_same_v<T, float>) {
        return v;
    } else if constexpr (std::is_same_v<T, llaisys::fp16_t>) {
        return llaisys::utils::_f16_to_f32(v);
    } else if constexpr (std::is_same_v<T, llaisys::bf16_t>) {
        return llaisys::utils::_bf16_to_f32(v);
    } else {
        return (float)v;
    }
}

template <typename T>
inline T float_to_val(float v) {
    if constexpr (std::is_same_v<T, float>) {
        return v;
    } else if constexpr (std::is_same_v<T, llaisys::fp16_t>) {
        return llaisys::utils::_f32_to_f16(v);
    } else if constexpr (std::is_same_v<T, llaisys::bf16_t>) {
        return llaisys::utils::_f32_to_bf16(v);
    } else {
        return (T)v;
    }
}

// --- Self Attention 核心计算模板 ---
template<typename T>
void self_attention_kernel(T *attn_val, const T *q, const T *k, const T *v,
                           size_t seqlen, size_t total_len, 
                           size_t nhead, size_t nkvhead, 
                           size_t d, size_t dv,
                           float scale) {
    
    // GQA 分组大小
    size_t group_size = nhead / nkvhead;
    
    // 预分配 score 数组 (用于存放 softmax 前后的值)
    // 每个线程或每次循环复用这个 buffer
    std::vector<float> scores(total_len);

    // 1. 遍历 Queries (Sequence Length)
    for (size_t t = 0; t < seqlen; ++t) {
        // 当前 Query Token 的全局位置
        // 假设 Q 对应 K/V 的最后 seqlen 个元素
        size_t current_global_pos = total_len - seqlen + t;

        // 2. 遍历 Heads
        for (size_t h = 0; h < nhead; ++h) {
            // 找到对应的 KV Head 索引 (GQA)
            size_t kv_h = h / group_size;

            // --- 步骤 A: 计算 Attention Scores (Q * K^T) ---
            float max_score = -std::numeric_limits<float>::infinity();

            for (size_t pos = 0; pos < total_len; ++pos) {
                // Causal Mask: 如果 Key 的位置超过了 Query 的位置，则不可见
                if (pos > current_global_pos) {
                    scores[pos] = -std::numeric_limits<float>::infinity();
                    continue;
                }

                // 点积: Q[t, h] dot K[pos, kv_h]
                float dot = 0.0f;
                
                // 指针定位
                // q: [seqlen, nhead, d]
                size_t q_offset = t * nhead * d + h * d;
                // k: [total_len, nkvhead, d]
                size_t k_offset = pos * nkvhead * d + kv_h * d;

                for (size_t i = 0; i < d; ++i) {
                    float q_val = val_to_float(q[q_offset + i]);
                    float k_val = val_to_float(k[k_offset + i]);
                    dot += q_val * k_val;
                }

                dot *= scale;
                scores[pos] = dot;
                
                if (dot > max_score) max_score = dot;
            }

            // --- 步骤 B: Softmax ---
            float sum_exp = 0.0f;
            for (size_t pos = 0; pos < total_len; ++pos) {
                if (scores[pos] == -std::numeric_limits<float>::infinity()) {
                    scores[pos] = 0.0f; // mask 掉的位置概率为 0
                } else {
                    float val = std::exp(scores[pos] - max_score);
                    scores[pos] = val;
                    sum_exp += val;
                }
            }
            // 归一化
            float inv_sum = 1.0f / sum_exp;
            for (size_t pos = 0; pos < total_len; ++pos) {
                scores[pos] *= inv_sum;
            }

            // --- 步骤 C: Weighted Sum (Scores * V) ---
            // 结果写入 attn_val[t, h] (维度 dv)
            size_t out_offset = t * nhead * dv + h * dv;
            
            // 先清零当前输出向量
            for (size_t i = 0; i < dv; ++i) {
                // 这里暂时用 float 累加提高精度
            }
            std::vector<float> acc(dv, 0.0f);

            for (size_t pos = 0; pos < total_len; ++pos) {
                float weight = scores[pos];
                if (weight == 0.0f) continue; // 优化：跳过 0 权重

                // v: [total_len, nkvhead, dv]
                size_t v_offset = pos * nkvhead * dv + kv_h * dv;
                
                for (size_t i = 0; i < dv; ++i) {
                    float v_val = val_to_float(v[v_offset + i]);
                    acc[i] += weight * v_val;
                }
            }

            // 存回结果
            for (size_t i = 0; i < dv; ++i) {
                attn_val[out_offset + i] = float_to_val<T>(acc[i]);
            }
        }
    }
}

// --- 入口分发 ---
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    size_t seqlen, size_t total_len, 
                    size_t nhead, size_t nkvhead, 
                    size_t d, size_t dv,
                    float scale,
                    llaisysDataType_t dtype) {
    
    if (dtype == LLAISYS_DTYPE_F32) {
        self_attention_kernel<float>(
            (float*)attn_val, (const float*)q, (const float*)k, (const float*)v,
            seqlen, total_len, nhead, nkvhead, d, dv, scale
        );
    } else if (dtype == LLAISYS_DTYPE_F16) {
        self_attention_kernel<llaisys::fp16_t>(
            (llaisys::fp16_t*)attn_val, (const llaisys::fp16_t*)q, (const llaisys::fp16_t*)k, (const llaisys::fp16_t*)v,
            seqlen, total_len, nhead, nkvhead, d, dv, scale
        );
    } else if (dtype == LLAISYS_DTYPE_BF16) {
        self_attention_kernel<llaisys::bf16_t>(
            (llaisys::bf16_t*)attn_val, (const llaisys::bf16_t*)q, (const llaisys::bf16_t*)k, (const llaisys::bf16_t*)v,
            seqlen, total_len, nhead, nkvhead, d, dv, scale
        );
    }
}

} // namespace llaisys::ops::cpu