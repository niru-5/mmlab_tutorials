python - <<'PY'
import torch, ctypes, ctypes.util, os
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))
print("find_library('cuda'):", ctypes.util.find_library("cuda"))
try:
    h = ctypes.CDLL("libcuda.so.1")
    print("Loaded lib:", h._name)
except OSError as e:
    print("libcuda load error:", e)
print("device_count:", torch.cuda.device_count())
try:
    torch.cuda._lazy_init()
    print("lazy_init OK")
except Exception as e:
    print("lazy_init exception:", repr(e))
PY
