#!/usr/bin/env python
"""
evaluate_oscnet.py

This script evaluates the PSNR and SSIM for OSCNet outputs.
It expects the following folder structure inside the results directory:

  results_dir/
     ├── gt          (ground truth images, e.g., "1.png", "2.png", ...)
     └── oscplus     (if evaluating OSCNet+ reconstructed images)
         or
         osc       (if evaluating OSCNet reconstructed images)

Depending on the model type provided via --model_type, the script
will output the results in either "metrics_adapted_oscplus.txt" (for OSCNet+)
or "metrics_adapted_osc.txt" (for OSCNet).

Usage examples:
    python evaluate_oscnet.py --results_dir ./save_results --model_type oscnetplus
    python evaluate_oscnet.py --results_dir ./save_results --model_type oscnet

Dependencies:
    numpy, matplotlib, scikit-image
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import argparse


def to_grayscale(image):
    if image.ndim >= 3 and image.shape[-1] > 1:
        return image[..., 0]
    return image


def dynamic_win_size(image, default=7):
    h, w = image.shape[:2]
    min_dim = min(h, w)
    if min_dim < 3:
        return None
    win = min(default, min_dim)
    if win % 2 == 0:
        win -= 1
    return win


def evaluate_oscnet_image(results_dir, model_type, output_file):
    """
    Evaluate PSNR and SSIM for OSCNet (or OSCNet+) outputs.

    Expected directory structure:
      results_dir/
         ├─ gt      (ground truth images)
         └─ [oscplus or osc]   (reconstructed images)

    The output filename is chosen based on the model type.
    """
    # Set the reconstructed image folder based on model_type
    if model_type.lower() == "oscnetplus":
        recon_folder = "oscplus"
        if not output_file:
            output_file = "metrics_adapted_oscplus.txt"
    elif model_type.lower() == "oscnet":
        recon_folder = "osc"
        if not output_file:
            output_file = "metrics_adapted_osc.txt"
    else:
        print("Invalid model_type. Choose either 'oscnetplus' or 'oscnet'.")
        return

    # Define folder paths
    gt_dir = os.path.join(results_dir, "gt")
    recon_dir = os.path.join(results_dir, recon_folder)

    # Check that these directories exist
    for d in [gt_dir, recon_dir]:
        if not os.path.isdir(d):
            print(f"Warning: directory does not exist => {d}")

    # Get common PNG filenames from both folders
    gt_files = {f for f in os.listdir(gt_dir) if f.lower().endswith(".png")}
    recon_files = {f for f in os.listdir(recon_dir) if f.lower().endswith(".png")}
    common_filenames = sorted(gt_files & recon_files)

    if not common_filenames:
        print("No common PNG files found in gt and reconstruction directories.")
        return

    psnr_vals, ssim_vals = [], []
    table_rows = []
    table_header = f"{'Filename':<25}{'PSNR':>12}{'SSIM':>12}\n" + "-" * 50

    for filename in common_filenames:
        gt_path = os.path.join(gt_dir, filename)
        recon_path = os.path.join(recon_dir, filename)

        if not os.path.exists(gt_path) or not os.path.exists(recon_path):
            print(f"Skipping {filename} - missing file.")
            continue

        # Load images and convert to grayscale if needed
        gt_img = to_grayscale(np.squeeze(plt.imread(gt_path).astype(np.float64)))
        recon_img = to_grayscale(np.squeeze(plt.imread(recon_path).astype(np.float64)))

        print(f"\nProcessing {filename}:")
        print(f"GT image range: [{gt_img.min():.3f}, {gt_img.max():.3f}], shape: {gt_img.shape}")
        print(f"Recon image range: [{recon_img.min():.3f}, {recon_img.max():.3f}], shape: {recon_img.shape}")

        win = dynamic_win_size(gt_img, default=7)
        if win is None:
            print(f"Skipping {filename} for SSIM due to small dimensions: {gt_img.shape}")
            continue

        # Determine the data range dynamically based on the images
        data_range = max(gt_img.max(), recon_img.max()) - min(gt_img.min(), recon_img.min())
        data_range = data_range if data_range > 0 else 1e-6

        psnr_val = psnr(gt_img, recon_img, data_range=data_range)
        ssim_val = ssim(gt_img, recon_img, data_range=data_range, win_size=win, channel_axis=None)

        psnr_vals.append(psnr_val)
        ssim_vals.append(ssim_val)

        row = f"{filename:<25}{psnr_val:>12.2f}{ssim_val:>12.4f}"
        table_rows.append(row)

    def mean_std_str(arr):
        return f"{np.mean(arr):.2f} ± {np.std(arr):.2f}" if arr else "0.00 ± 0.00"

    summary = (
        "\nSummary (mean ± std)\n"
        f"PSNR: {mean_std_str(psnr_vals)}\n"
        f"SSIM: {np.mean(ssim_vals):.4f} ± {np.std(ssim_vals):.4f}\n"
    )

    full_report = (
            f"{model_type.upper()} Image Quality Evaluation\n"
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Results Directory: {os.path.abspath(results_dir)}\n\n"
            + table_header + "\n" +
            "\n".join(table_rows) +
            "\n" + summary
    )

    print("\n" + full_report)
    with open(output_file, 'w') as f:
        f.write(full_report)

    print(f"\nReport saved to: {os.path.abspath(output_file)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate OSCNet image quality metrics")
    parser.add_argument("--results_dir", type=str, default="./save_results", help="Directory containing the results")
    parser.add_argument("--model_type", type=str, default="oscnetplus", choices=["oscnetplus", "oscnet"],
                        help="Specify 'oscnetplus' (default) or 'oscnet'")
    parser.add_argument("--output_file", type=str, default="", help="Filename for the output report (optional)")
    args = parser.parse_args()

    evaluate_oscnet_image(args.results_dir, args.model_type, args.output_file)
