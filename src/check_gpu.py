import torch
import sys

def check_gpu():
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"\n✅ CUDA is available!")
        print(f"Device Verification: {torch.cuda.get_device_name(0)}")
        print(f"Number of Devices: {torch.cuda.device_count()}")
    else:
        print(f"\n❌ CUDA is NOT available.")
        print("Running on CPU.")

if __name__ == "__main__":
    check_gpu()
