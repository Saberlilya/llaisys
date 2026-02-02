#include "op.hpp"
#include "cpu/argmax_cpu.hpp" // 引入我们马上要写的 CPU 头文件

namespace llaisys::ops {

// 任务要求：获取张量 vals 的最大值及其索引
// max_idx: 存储索引的结果张量 (通常是 Int64 或 Int32)
// max_val: 存储最大值的结果张量 (类型与 vals 一致)
// vals: 输入张量 (1D)
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    // 1. 检查设备：只处理 CPU
    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        // 2. 获取元素总数 (假设 vals 是 1D，直接取 numel)
        size_t numel = vals->numel();
        
        // 3. 调用 CPU 实现
        return cpu::argmax(
            max_idx->data(), 
            max_val->data(), 
            vals->data(), 
            numel,
            vals->dtype(),    // 输入数据的类型
            max_idx->dtype()  // 索引数据的类型
        );
    }
    
    throw std::runtime_error("Argmax: device not supported");
}

} // namespace llaisys::ops