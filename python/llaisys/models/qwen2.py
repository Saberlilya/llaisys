import json
import mmap
import struct
from pathlib import Path
from typing import Sequence, List

import numpy as np
from ctypes import (
    POINTER,
    c_int,
    c_int64,
    c_size_t,
    c_void_p,
    memmove,
    byref,
)

from ..libllaisys import (
    LIB_LLAISYS,
    DeviceType,
    DataType,
    llaisysDeviceType_t,
)
from ..libllaisys.qwen2 import LlaisysQwen2Meta, LlaisysQwen2Weights
from ..tensor import Tensor


class Qwen2:
    """
    Python wrapper for Qwen2 model inference implemented in C/C++ backend.
    This file must NOT implement inference with PyTorch or any other Python frameworks.
    """

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        self.model_path = Path(model_path)
        self.device = device
        self._tensor_refs = []  # keep Tensor objects alive

        # tensorGetData is needed to copy numpy data into backend tensor
        if not hasattr(LIB_LLAISYS, "tensorGetData"):
            raise RuntimeError(
                "LIB_LLAISYS.tensorGetData not found. "
                "Your backend tensor API should expose tensorGetData for weight loading."
            )
        LIB_LLAISYS.tensorGetData.restype = c_void_p

        # ---- Load config.json ----
        config_path = self.model_path / "config.json"
        if not config_path.exists():
            candidates = list(self.model_path.rglob("config.json"))
            if candidates:
                config_path = candidates[0]
            else:
                raise FileNotFoundError("config.json not found under model_path")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.nlayer = int(config["num_hidden_layers"])
        hs = int(config["hidden_size"])
        nh = int(config["num_attention_heads"])
        nkvh = int(config.get("num_key_value_heads", nh))
        dh = hs // nh
        di = int(config["intermediate_size"])
        maxseq = int(config.get("max_position_embeddings", 2048))
        voc = int(config["vocab_size"])
        epsilon = float(config["rms_norm_eps"])
        theta = float(config.get("rope_theta", 1000000.0))
        end_token = int(config.get("eos_token_id", 151643))
        self.end_token = end_token

        # Assignment#3 baseline: load weights as F32 into LLAISYS tensors
        meta = LlaisysQwen2Meta()
        meta.dtype = DataType.F32.value
        meta.nlayer = self.nlayer
        meta.hs = hs
        meta.nh = nh
        meta.nkvh = nkvh
        meta.dh = dh
        meta.di = di
        meta.maxseq = maxseq
        meta.voc = voc
        meta.epsilon = epsilon
        meta.theta = theta
        meta.end_token = end_token

        # Create model backend
        self._model_handle = LIB_LLAISYS.llaisysQwen2ModelCreate(
            byref(meta),
            llaisysDeviceType_t(self.device.value),
            None,
            0,
        )
        if not self._model_handle:
            raise RuntimeError("Failed to create Qwen2 model in backend.")

        # Get weights struct (backend should allocate pointer arrays for per-layer weights)
        wptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model_handle)
        if not wptr:
            raise RuntimeError("Failed to get Qwen2 weights struct pointer from backend.")
        self._weights = wptr.contents

        # Load safetensors into weights
        self._load_weights()

    # -----------------------------
    # Weight loading: safetensors
    # -----------------------------
    def _load_weights(self):
        files = sorted(self.model_path.glob("*.safetensors"))
        if not files:
            files = sorted(self.model_path.rglob("*.safetensors"))
        if not files:
            raise FileNotFoundError("No .safetensors files found under model_path")

        for file in files:
            self._load_safetensors_file(file)

    def _load_safetensors_file(self, file_path: Path):
        with open(file_path, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_size).decode("utf-8"))
            data_start = 8 + header_size

            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                for name, info in header.items():
                    if name == "__metadata__":
                        continue
                    dtype_str = info["dtype"]
                    offsets = info["data_offsets"]
                    shape = info["shape"]

                    start = data_start + offsets[0]
                    end = data_start + offsets[1]
                    raw_bytes = mm[start:end]

                    # We load as float32 for now (Assignment#3 correctness first)
                    arr_f32 = self._to_float32(raw_bytes, dtype_str)
                    if arr_f32 is None:
                        continue

                    self._assign_weight(name, arr_f32, shape, DataType.F32)

    @staticmethod
    def _to_float32(raw_bytes: bytes, dtype_str: str):
        if dtype_str == "BF16":
            raw_u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
            # BF16 -> FP32 by placing bf16 in high 16 bits
            arr_f32 = (raw_u16.astype(np.uint32) << 16).view(np.float32)
            return arr_f32
        if dtype_str == "F16":
            return np.frombuffer(raw_bytes, dtype=np.float16).astype(np.float32)
        if dtype_str == "F32":
            return np.frombuffer(raw_bytes, dtype=np.float32)
        # ignore INT types etc.
        return None

    def _new_tensor_and_copy(self, shape: List[int], dtype: DataType, data_f32: np.ndarray):
        t = Tensor(shape=shape, dtype=dtype, device=self.device)

        # Get internal handle (Tensor wrapper variants)
        handle = getattr(t, "_tensor", None)
        if handle is None:
            handle = getattr(t, "_handle", None)
        if handle is None:
            handle = getattr(t, "_impl", None)
        if handle is None:
            raise RuntimeError("Cannot locate internal llaisysTensor handle inside Tensor object.")

        dst_ptr = LIB_LLAISYS.tensorGetData(handle)
        if not dst_ptr:
            raise RuntimeError("tensorGetData returned null pointer.")
        memmove(dst_ptr, data_f32.ctypes.data, data_f32.nbytes)

        # keep alive
        self._tensor_refs.append(t)
        return handle

    def _assign_weight(self, name: str, data: np.ndarray, shape: List[int], dtype: DataType):
        # create LLAISYS tensor and copy data
        handle = self._new_tensor_and_copy(shape, dtype, data)

        w = self._weights

        # top-level
        if name == "model.embed_tokens.weight":
            w.in_embed = handle
            return
        if name == "lm_head.weight":
            w.out_embed = handle
            return
        if name == "model.norm.weight":
            w.out_norm_w = handle
            return

        # per-layer
        if not name.startswith("model.layers."):
            return

        parts = name.split(".")
        # model.layers.{idx}.{module}.(submodule.)weight
        try:
            idx = int(parts[2])
        except Exception:
            return

        # We only map weights needed for inference
        # Ignore biases if they don't exist; header has *_b pointers but many Qwen2 use bias=False
        # We'll map if name matches.
        if idx < 0 or idx >= self.nlayer:
            return

        module = parts[3]

        # input_layernorm.weight
        if module == "input_layernorm" and parts[-1] == "weight":
            w.attn_norm_w[idx] = handle
            return

        # post_attention_layernorm.weight
        if module == "post_attention_layernorm" and parts[-1] == "weight":
            w.mlp_norm_w[idx] = handle
            return

        # self_attn.{q/k/v/o}_proj.weight (and optional bias)
        if module == "self_attn":
            if len(parts) < 6:
                return
            sub = parts[4]  # q_proj / k_proj / v_proj / o_proj
            last = parts[-1]  # weight or bias

            if sub == "q_proj" and last == "weight":
                w.attn_q_w[idx] = handle
                return
            if sub == "k_proj" and last == "weight":
                w.attn_k_w[idx] = handle
                return
            if sub == "v_proj" and last == "weight":
                w.attn_v_w[idx] = handle
                return
            if sub == "o_proj" and last == "weight":
                w.attn_o_w[idx] = handle
                return

            # optional bias
            if sub == "q_proj" and last == "bias":
                w.attn_q_b[idx] = handle
                return
            if sub == "k_proj" and last == "bias":
                w.attn_k_b[idx] = handle
                return
            if sub == "v_proj" and last == "bias":
                w.attn_v_b[idx] = handle
                return

            return

        # mlp.{gate/up/down}_proj.weight
        if module == "mlp":
            if len(parts) < 6:
                return
            sub = parts[4]
            if parts[-1] != "weight":
                return
            if sub == "gate_proj":
                w.mlp_gate_w[idx] = handle
                return
            if sub == "up_proj":
                w.mlp_up_w[idx] = handle
                return
            if sub == "down_proj":
                w.mlp_down_w[idx] = handle
                return

    # -----------------------------
    # Generation (argmax baseline)
    # -----------------------------
    def generate(self, inputs: Sequence[int], max_new_tokens: int = 128, **kwargs) -> List[int]:
        """
        Argmax generation for assignment test (--test sets top_k=1 top_p=1 temperature=1).
        Contract with backend:
          - Infer(model, token_ids, ntoken) consumes new tokens, updates KV cache,
            and returns next token id (argmax).
        """
        tokens = list(int(x) for x in inputs)

        if len(tokens) == 0:
            raise ValueError("inputs must be non-empty")

        # 1) prefill with full prompt
        c_prompt = (c_int64 * len(tokens))(*tokens)
        next_tok = int(
            LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model_handle,
                c_prompt,
                c_size_t(len(tokens)),
            )
        )
        tokens.append(next_tok)

        # 2) decode step-by-step (feed last token each time)
        for _ in range(max_new_tokens - 1):
            if tokens[-1] == self.end_token:
                break
            last = tokens[-1]
            c_last = (c_int64 * 1)(last)
            next_tok = int(
                LIB_LLAISYS.llaisysQwen2ModelInfer(
                    self._model_handle,
                    c_last,
                    c_size_t(1),
                )
            )
            tokens.append(next_tok)

        return tokens

    def __del__(self):
        try:
            if hasattr(self, "_model_handle") and self._model_handle:
                LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model_handle)
                self._model_handle = None
        except Exception:
            pass
