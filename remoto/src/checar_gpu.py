"""Usado pelo instalador: sai com 0 se o torch usa a GPU de verdade."""
import sys

try:
    import torch
    if not torch.cuda.is_available():
        sys.exit(1)
    a = torch.randn(256, 256, device='cuda', dtype=torch.float16)
    float((a @ a).sum())
    print('       torch %s | %s' % (torch.__version__, torch.cuda.get_device_name(0)))
except Exception:
    sys.exit(1)
