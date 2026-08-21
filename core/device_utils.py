import os
import torch

def get_optimal_device(force_cpu: bool = False, verbose: bool = True) -> str:
    """
    Determines the best available device for PyTorch execution.
    Checks CUDA capability compatibility to safely handle modern Blackwell (CC 12.0)
    or other next-gen GPUs under various PyTorch wheel configurations.
    """
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    if force_cpu or not torch.cuda.is_available():
        return "cpu"

    try:
        cap = torch.cuda.get_device_capability(0)
        # Standard PyTorch cu126 builds support CC up to 9.0.
        # cu128 / cu129 / cu130 / cu132 builds add CC 10.0, 11.0, 12.0 (sm_120).
        if cap[0] > 9:
            if verbose:
                print(f"[Device Manager] Found GPU with CUDA capability {cap[0]}.{cap[1]}. Utilizing CPU fallback unless cu129+ is installed.")
            return "cpu"

        # Verify allocation
        test_t = torch.zeros(1, device="cuda")
        _ = test_t + 1.0
        return "cuda"
    except Exception as e:
        if verbose:
            print(f"[Device Manager] CUDA initialization note: {e}. Falling back to CPU.")
        return "cpu"
