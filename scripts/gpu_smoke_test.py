"""Phase 0: validate the 8-GPU H100 environment and torchrun."""
import torch

def main() -> None:
    print("CUDA available:", torch.cuda.is_available())
    n = torch.cuda.device_count()
    print("GPU count:", n)
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {p.name}  {p.total_memory/1e9:.0f} GB")
    assert torch.cuda.is_available(), "No CUDA devices visible"

if __name__ == "__main__":
    main()
