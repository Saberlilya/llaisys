#include "op.hpp"
#include "cpu/swiglu_cpu.hpp"

namespace llaisys::ops {

// SwiGLU: out = up * SiLU(gate)
// 这是一个逐元素操作，形状完全一致
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        // 因为是逐元素运算，我们只关心总元素个数，不需要关心具体维度
        size_t numel = out->numel();

        return cpu::swiglu(
            out->data(),
            gate->data(),
            up->data(),
            numel,
            out->dtype()
        );
    }
    throw std::runtime_error("SwiGLU: device not supported");
}

} // namespace llaisys::ops