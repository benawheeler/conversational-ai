import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
    # Simple tensor test
    x = torch.rand(3, 3).cuda()
    print("Tensor on GPU:", x)
else:
    print("No CUDA device detected. If you have an NVIDIA GPU, install the correct PyTorch CUDA wheel.") 