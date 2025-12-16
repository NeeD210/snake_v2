# Snake AI on Google Colab

This guide consolidates all steps to run the Snake AI training on Google Colab efficiently using a T4 GPU.

## 1. Preparation
Ensure your local `snake_ai/main.py` has the following crucial adaptations for Colab:
1.  **GPU Device Handling**: Explicitly sending models and tensors to `device`.
2.  **Monitoring**: The `starmap_async` loop with `psutil`/`nvidia-smi` logging.
3.  **Process Path**: Use `/usr/bin/nvidia-smi` instead of just `nvidia-smi` to avoid `FileNotFoundError`.

**Zip the project**:
```powershell
Compress-Archive -Path snake_ai -DestinationPath snake_ai_colab_v3.zip -Force
```

## 2. Colab Setup
1.  **New Notebook**: Create a new notebook at [colab.research.google.com](https://colab.research.google.com/).
2.  **Runtime**: Go to `Runtime` > `Change runtime type` > Select **T4 GPU**.
3.  **Upload**: Upload your `snake_ai_colab_v3.zip` to the session storage.

## 3. Execution
Run these commands in a notebook cell:

```python
# 1. Unzip
!unzip -o snake_ai_colab_v3.zip

# 2. Run Training
# Workers: 4 is the recommended starting point for T4 (2 vCPU).
# If CPU is pinned at 100%, lower this to 2 or 3.
!python snake_ai/main.py --workers 4
```

## 4. Performance Tuning
Colab's T4 instance has a strong GPU but a weak CPU (2 cores).
*   **Goal**: Keep GPU usage > 80% without pinning CPU to 100%.
*   **Monitoring**: The script prints a status line every 5s:
    `[Monitor] CPU: 95.0% | GPU Util: 15% | GPU Mem: 600MiB`

| Observation | Diagnosis | Action |
| :--- | :--- | :--- |
| **High CPU (100%), Low GPU (<30%)** | **CPU Bottleneck**. The 2 cores can't feed the GPU fast enough. | **Decrease workers** (Try 2 or 3). |
| **Low CPU (<60%), Low GPU (<30%)** | **Underutilization**. System is waiting. | **Increase workers** (Try 6 or 8). |
| **High GPU (>80%)** | **Sweet Spot**. You are getting maximum value. | Keep settings. |

## 5. Troubleshooting
*   **`FileNotFoundError: 'nvidia-smi'`**: You must use the full path `/usr/bin/nvidia-smi` in `main.py`.
*   **`TypeError: can't convert cuda:0...`**: In `mcts.py`, ensure you move tensors to CPU before numpy conversion: `p = torch.exp(p).squeeze().cpu().numpy()`.
*   **OOM (Out Of Memory)**: If the GPU runs out of memory, reduce workers (each worker takes ~500MB VRAM).
