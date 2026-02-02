import json
import mmap
import struct
import numpy as np
from pathlib import Path
from typing import Sequence, List
from ctypes import (
    Structure, byref, c_int, c_int64, c_size_t, c_float, 
    cast, POINTER, c_void_p, memmove
)

from ..libllaisys import LIB_LLAISYS, DeviceType, DataType, llaisysTensor_t, llaisysDeviceType_t
from ..tensor import Tensor

class LlaisysQwen2Weights(Structure):
    _fields_ = [
        ("in_embed", llaisysTensor_t),
        ("out_embed", llaisysTensor_t),
        ("out_norm_w", llaisysTensor_t),
        ("attn_norm_w", POINTER(llaisysTensor_t)),
        ("attn_q_w", POINTER(llaisysTensor_t)),
        ("attn_q_b", POINTER(llaisysTensor_t)),
        ("attn_k_w", POINTER(llaisysTensor_t)),
        ("attn_k_b", POINTER(llaisysTensor_t)),
        ("attn_v_w", POINTER(llaisysTensor_t)),
        ("attn_v_b", POINTER(llaisysTensor_t)),
        ("attn_o_w", POINTER(llaisysTensor_t)),
        ("mlp_norm_w", POINTER(llaisysTensor_t)),
        ("mlp_gate_w", POINTER(llaisysTensor_t)),
        ("mlp_up_w", POINTER(llaisysTensor_t)),
        ("mlp_down_w", POINTER(llaisysTensor_t)),
    ]

class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        self.model_path = Path(model_path)
        self.device = device
        self._tensor_refs = []
        
        # 定义 Raw Create 函数签名 (必须匹配 C++)
        LIB_LLAISYS.llaisysQwen2ModelCreateRaw.restype = c_void_p
        LIB_LLAISYS.llaisysQwen2ModelCreateRaw.argtypes = [
            c_size_t, c_size_t, c_size_t, c_size_t, c_size_t, c_size_t, 
            c_size_t, c_size_t, c_float, c_float, c_int, c_int,         
            llaisysDeviceType_t, POINTER(c_int), c_int                  
        ]
        
        if hasattr(LIB_LLAISYS, "tensorGetData"):
            LIB_LLAISYS.tensorGetData.restype = c_void_p
            LIB_LLAISYS.tensorGetData.argtypes = [llaisysTensor_t]

        LIB_LLAISYS.llaisysQwen2ModelWeights.restype = POINTER(LlaisysQwen2Weights)
        LIB_LLAISYS.llaisysQwen2ModelWeights.argtypes = [c_void_p]

        # Config
        config_path = self.model_path / "config.json"
        if not config_path.exists():
             candidates = list(self.model_path.rglob("config.json"))
             if candidates: config_path = candidates[0]
             else: raise FileNotFoundError(f"config.json not found")

        with open(config_path, "r") as f:
            config = json.load(f)
            
        nlayer = config["num_hidden_layers"]
        hs = config["hidden_size"]
        nh = config["num_attention_heads"]
        nkvh = config.get("num_key_value_heads", nh)
        dh = hs // nh
        di = config["intermediate_size"]
        maxseq = config.get("max_position_embeddings", 2048)
        voc = config["vocab_size"]
        epsilon = config["rms_norm_eps"]
        theta = config.get("rope_theta", 1000000.0)
        end_token = config.get("eos_token_id", 151643)
        dtype_val = DataType.F32.value
        self.end_token = end_token
        self.nlayer = nlayer # 保存供 check

        print(f"[Python] Creating Model via Raw API (nlayer={nlayer})...")
        self._model_handle = LIB_LLAISYS.llaisysQwen2ModelCreateRaw(
            nlayer, hs, nh, nkvh, dh, di, maxseq, voc, epsilon, theta, end_token, dtype_val,
            llaisysDeviceType_t(self.device.value), None, 0
        )
        if not self._model_handle:
            raise RuntimeError("Failed to create C++ model handle")
        
        self._weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model_handle).contents
        self._load_weights()

    def _load_weights(self):
        print("[Python] Loading weights files...")
        files = sorted(self.model_path.glob("*.safetensors"))
        if not files: files = sorted(self.model_path.rglob("*.safetensors"))
        for file in files: self._load_safetensors_file(file)
        print("[Python] Weights loaded successfully.")

    def _load_safetensors_file(self, file_path):
        with open(file_path, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_size).decode("utf-8"))
            data_start = 8 + header_size
            
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                for name, info in header.items():
                    if name == "__metadata__": continue
                    dtype_str = info["dtype"]
                    offsets = info["data_offsets"]
                    tensor_shape = info["shape"]
                    start = data_start + offsets[0]
                    end = data_start + offsets[1]
                    
                    raw_bytes = mm[start:end]
                    
                    arr_f32 = None
                    if dtype_str == "BF16":
                        raw_u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
                        arr_f32 = (raw_u16.astype(np.uint32) << 16).view(np.float32)
                    elif dtype_str == "F16":
                        arr_f32 = np.frombuffer(raw_bytes, dtype=np.float16).astype(np.float32)
                    elif dtype_str == "F32":
                        arr_f32 = np.frombuffer(raw_bytes, dtype=np.float32)
                    else: continue

                    self._assign_weight_manual(name, arr_f32, tensor_shape, DataType.F32)

    def _assign_weight_manual(self, name: str, data: np.ndarray, shape: List[int], dtype: DataType):
        t = Tensor(shape=shape, dtype=dtype, device=self.device)
        
        handle = getattr(t, "_tensor", None)
        if handle is None: handle = getattr(t, "_handle", None)
        if handle is None: handle = getattr(t, "_impl", None)
        if handle is None: 
            for attr in dir(t): 
                if attr.startswith("_") and "LlaisysTensor" in str(type(getattr(t, attr))): handle = getattr(t, attr)
        if handle is None: return

        memmove(LIB_LLAISYS.tensorGetData(handle), data.ctypes.data, data.nbytes)
        self._tensor_refs.append(t)

        w = self._weights
        if name == "model.embed_tokens.weight": w.in_embed = handle
        elif name == "lm_head.weight": w.out_embed = handle
        elif name == "model.norm.weight": w.out_norm_w = handle
        elif name.startswith("model.layers."):
            parts = name.split(".")
            idx = int(parts[2])
            module = parts[3]
            if parts[-1] != "weight": return
            if module == "input_layernorm": w.attn_norm_w[idx] = handle
            elif module == "post_attention_layernorm": w.mlp_norm_w[idx] = handle
            elif module == "self_attn":
                sub = parts[4]
                if sub == "q_proj": w.attn_q_w[idx] = handle
                elif sub == "k_proj": w.attn_k_w[idx] = handle
                elif sub == "v_proj": w.attn_v_w[idx] = handle
                elif sub == "o_proj": w.attn_o_w[idx] = handle
            elif module == "mlp":
                sub = parts[4]
                if sub == "gate_proj": w.mlp_gate_w[idx] = handle
                elif sub == "up_proj": w.mlp_up_w[idx] = handle
                elif sub == "down_proj": w.mlp_down_w[idx] = handle

    def generate(self, inputs: Sequence[int], max_new_tokens: int = 100, **kwargs) -> List[int]:
        tokens = list(inputs)
        c_tokens = (c_int64 * len(tokens))(*tokens)
        LIB_LLAISYS.llaisysQwen2ModelPrefill(self._model_handle, c_tokens, c_size_t(len(tokens)))
        return tokens
    
    def __del__(self):
        if hasattr(self, "_model_handle") and self._model_handle:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model_handle)