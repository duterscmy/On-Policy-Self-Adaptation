#!/usr/bin/env python
import platform, sys
print("Python:", sys.version)
print("Platform:", platform.platform())
print("Machine:", platform.machine())
try:
    import torch
except Exception as e:
    print("PyTorch import FAILED:", repr(e)); raise SystemExit(1)
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available(): raise SystemExit(1)
print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {p.name}, {p.total_memory/1024**3:.1f} GiB, capability {p.major}.{p.minor}")
print("BF16 supported:", torch.cuda.is_bf16_supported())
