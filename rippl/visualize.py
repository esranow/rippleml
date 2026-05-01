import os
import sys
import zipfile
import json
import tempfile
import numpy as np

def visualize(rpx_path: str):
    if not os.path.exists(rpx_path):
        print(f"File not found: {rpx_path}")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(rpx_path, 'r') as zipf:
            zipf.extractall(temp_dir)

        diag_dir = os.path.join(temp_dir, "diagnostics")
        metrics_path = os.path.join(diag_dir, "metrics.json")
        residuals_path = os.path.join(diag_dir, "residuals.npy")

        metrics = {}
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)

        try:
            import matplotlib.pyplot as plt
            has_mpl = True
        except ImportError:
            has_mpl = False

        if not has_mpl:
            print(json.dumps(metrics, indent=2))
            sys.exit(0)

        # Matplotlib is available, render heatmap
        if os.path.exists(residuals_path):
            res = np.load(residuals_path)
            plt.figure()
            if res.ndim > 1:
                plt.imshow(res, cmap='inferno', aspect='auto')
                plt.colorbar()
            else:
                plt.plot(res)
            plt.title("Diagnostics")
            plt.savefig("heatmap.png")
            print("Saved heatmap.png")
        else:
            print(json.dumps(metrics, indent=2))
