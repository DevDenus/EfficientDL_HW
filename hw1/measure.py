import gc
import torch
import numpy as np
import pandas as pd
import pynvml
from tqdm import tqdm

from models import CNNModel

PERFORMANCE_TRIES = 50

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

# My WSL2 is using Unified Memory, so we have to restrict using RAM for inference
# in order to be able to catch OOM(I had about 63 GB RAM+VRAM available before)
torch.cuda.set_per_process_memory_fraction(0.85, 0)

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def gpu_energy_mj():
    return pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)

# Took bigger model than described
model = CNNModel(256, 100).cuda().eval()

def benchmark_model_performance(image_size : int, batch_size : int):
    try:
        # Clear cache and collect garbage, trying to avoid extra in-flight reallocations
        gc.collect()
        torch.cuda.empty_cache()

        test_image = torch.rand((batch_size, 3, image_size, image_size), dtype=torch.float32, device="cuda")

        # Warmup
        with torch.inference_mode():
            model(test_image)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        latencies = []
        e0 = gpu_energy_mj()

        for _ in range(PERFORMANCE_TRIES):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            with torch.inference_mode():
                model(test_image)

            end.record()
            end.synchronize()
            latencies.append(start.elapsed_time(end))

        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated()
        e1 = gpu_energy_mj()
        energy_mj = (e1 - e0) / PERFORMANCE_TRIES
        median_latency = np.median(latencies)

        return peak_memory, median_latency, energy_mj

    except torch.cuda.OutOfMemoryError:
        print("OOM occured")
        return "OOM", None, None


result = {
    key : [] for key in ("S", "B", "latency_ms", "memory_bytes", "energy_mJ", "is_validation")
}

images_sizes = [32, 64, 128, 224, 256, 384, 512, 768, 1024]
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512]
image_batch_paired = [
    (image_size, batch_size) for image_size in images_sizes for batch_size in batch_sizes
]
for image_size, batch_size in tqdm(image_batch_paired):
    peak_memory, median_latency, energy_mj = benchmark_model_performance(image_size, batch_size)

    result["S"].append(image_size)
    result["B"].append(batch_size)
    result["memory_bytes"].append(peak_memory)
    result["latency_ms"].append(median_latency)
    result["energy_mJ"].append(energy_mj)
    result["is_validation"].append(False)



val_images_size = [112, 192, 336, 448, 800]
val_batch_sizes = [59, 101, 217, 417]
val_image_batch_paired = [
    (image_size, batch_size) for image_size in val_images_size for batch_size in val_batch_sizes
]
for image_size, batch_size in tqdm(val_image_batch_paired):
    peak_memory, median_latency, energy_mj = benchmark_model_performance(image_size, batch_size)

    result["S"].append(image_size)
    result["B"].append(batch_size)
    result["memory_bytes"].append(peak_memory)
    result["latency_ms"].append(median_latency)
    result["energy_mJ"].append(energy_mj)
    result["is_validation"].append(True)

result_df = pd.DataFrame.from_dict(result)
result_df.to_csv("hw1/results/measurements.csv")
