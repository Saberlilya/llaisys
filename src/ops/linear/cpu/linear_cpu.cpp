#include "linear_cpu.hpp"
#include "../../../utils.hpp" // 包含类型转换工具
#include <type_traits>
#include <vector>

namespace llaisys::ops::cpu {

// --- 辅助 1：把任意类型转为 float (用于计算) ---
template <typename T>
inline float val_to_float(T v) {
    if constexpr (std::is_same_v<T, float>) {
        return v;
    } else if constexpr (std::is_same_v<T, llaisys::fp16_t>) {
        return llaisys::utils::_f16_to_f32(v);
    } else if constexpr (std::is_same_v<T, llaisys::bf16_t>) {
        return llaisys::utils::_bf16_to_f32(v); // 如果报错，尝试用 _bf16_to_float
    } else {
        return (float)v;
    }
}

// --- 辅助 2：把 float 转回任意类型 (用于存结果) ---
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

// --- 核心矩阵乘法模板 ---
template<typename T>
void linear_kernel(T *out, const T *in, const T *weight, const T *bias,
                   size_t M, size_t K, size_t N) {
    // 遍历 M 行 (Batch * Seq)
    for (size_t m = 0; m < M; ++m) {
        // 遍历 N 列 (Output Features)
        for (size_t n = 0; n < N; ++n) {
            
            float sum = 0.0f;
            // 遍历 K (Input Features) 进行点积
            // 公式：sum = sum(in[m, k] * weight[n, k])
            // 注意：weight 形状是 [N, K]，所以取 weight[n, k] 正好对应矩阵 W^T 的第 n 列
            for (size_t k = 0; k < K; ++k) {
                float x_val = val_to_float(in[m * K + k]);
                float w_val = val_to_float(weight[n * K + k]);
                sum += x_val * w_val;
            }
            
            // 加上偏置 (如果存在)
            if (bias) {
                sum += val_to_float(bias[n]);
            }
            
            // 存回结果 (转回对应类型 T)
            out[m * N + n] = float_to_val<T>(sum);
        }
    }
}

// --- 入口分发函数 ---
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            size_t M, size_t K, size_t N, llaisysDataType_t dtype) {
            
    // 1. Float32
    if (dtype == LLAISYS_DTYPE_F32) {
        linear_kernel<float>(
            (float*)out, (const float*)in, (const float*)weight, (const float*)bias, M, K, N
        );
    }
    // 2. Float16
    else if (dtype == LLAISYS_DTYPE_F16) {
        linear_kernel<llaisys::fp16_t>(
            (llaisys::fp16_t*)out, (const llaisys::fp16_t*)in, (const llaisys::fp16_t*)weight, 
            (const llaisys::fp16_t*)bias, M, K, N
        );
    }
    // 3. BFloat16
    else if (dtype == LLAISYS_DTYPE_BF16) {
        linear_kernel<llaisys::bf16_t>(
            (llaisys::bf16_t*)out, (const llaisys::bf16_t*)in, (const llaisys::bf16_t*)weight, 
            (const llaisys::bf16_t*)bias, M, K, N
        );
    }
}

} // namespace llaisys::ops::cpu