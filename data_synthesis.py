import os
import cv2
import numpy as np
import random
from tqdm import tqdm
from glob import glob

# ============================================================
# Path settings
# ============================================================
input_base = r"D:\dataset\imageses\image"       # Input image folder
output_base = r"D:\dataset\images_occlusion_lowlight"  # Output root directory
os.makedirs(output_base, exist_ok=True)


# ============================================================
# ---------- Occlusion augmentation ----------
# ============================================================

def simulate_occlusion_alpha(img, occ_size_ratio, alpha=0.5, mode="noise"):
    h, w, c = img.shape
    occ_h = int(h * occ_size_ratio)
    occ_w = int(w * occ_size_ratio)

    # Random position for occlusion block
    x1 = random.randint(0, max(0, w - occ_w))
    y1 = random.randint(0, max(0, h - occ_h))
    x2, y2 = x1 + occ_w, y1 + occ_h

    overlay = img.copy()
    if mode == "noise":
        block = np.random.randint(0, 256, (occ_h, occ_w, c), dtype=np.uint8)
    elif mode == "gray":
        block = np.full((occ_h, occ_w, c), 128, dtype=np.uint8)
    else:
        block = np.zeros((occ_h, occ_w, c), dtype=np.uint8)

    overlay[y1:y2, x1:x2] = (alpha * block + (1 - alpha) * img[y1:y2, x1:x2]).astype(np.uint8)
    return overlay


def process_occlusion_dataset():
    img_files = [f for f in os.listdir(input_base) if f.endswith(".jpg") or f.endswith(".png")]
    total_imgs = len(img_files)
    print(f"\nFound {total_imgs} images for occlusion processing")

    occlusion_ratios = [0.1, 0.2, 0.3]
    alpha_gradients = [0.25, 0.5, 0.75, 1.0]
    alpha_ratios = [0.25, 0.25, 0.25, 0.25]

    indices = np.arange(total_imgs)
    np.random.shuffle(indices)

    counts = [int(total_imgs * r) for r in alpha_ratios]
    counts[-1] = total_imgs - sum(counts[:-1])

    split_indices = []
    start = 0
    for count in counts:
        split_indices.append(indices[start:start + count])
        start += count

    for occ_idx, occ_ratio in enumerate(occlusion_ratios):
        print(f"\n==== Generating occ_l{occ_idx+1} (occlusion ratio {occ_ratio}) ====")
        save_dir = os.path.join(output_base, f"occ_l{occ_idx+1}")
        os.makedirs(save_dir, exist_ok=True)

        for i, alpha_val in enumerate(alpha_gradients):
            for idx in tqdm(split_indices[i], desc=f"occlusion_{occ_idx+1} alpha={alpha_val}"):
                img_path = os.path.join(input_base, img_files[idx])
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img_occ = simulate_occlusion_alpha(img, occ_ratio, alpha=alpha_val, mode="noise")
                save_path = os.path.join(save_dir, img_files[idx])
                cv2.imwrite(save_path, img_occ)


# ============================================================
# ---------- Low-light augmentation ----------
# ============================================================

def gamma_correction(img, gamma):
    img_norm = img / 255.0
    img_gamma = np.power(img_norm, gamma)
    return np.uint8(img_gamma * 255)


def adjust_contrast_brightness(img, alpha, beta):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def adjust_saturation(img, alpha_s):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= alpha_s
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def add_gaussian_noise(img, sigma):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.uint8(np.clip(noisy, 0, 255))


def simulate_lowlight(img, gamma, alpha, beta, alpha_s, sigma):
    out = gamma_correction(img, gamma)
    out = adjust_contrast_brightness(out, alpha, beta)
    out = adjust_saturation(out, alpha_s)
    out = add_gaussian_noise(out, sigma)
    return out


def process_lowlight_dataset():
    img_paths = glob(os.path.join(input_base, "*.jpg")) + glob(os.path.join(input_base, "*.png"))
    print(f"\nFound {len(img_paths)} images for low-light processing")

    lowlight_settings = [
        {"gamma": 0.6, "alpha": 0.7, "beta": -20, "alpha_s": 0.8, "sigma": 5},
        {"gamma": 0.4, "alpha": 0.5, "beta": -40, "alpha_s": 0.6, "sigma": 10},
        {"gamma": 0.3, "alpha": 0.4, "beta": -50, "alpha_s": 0.4, "sigma": 15},
    ]

    for i, setting in enumerate(lowlight_settings, 1):
        save_dir = os.path.join(output_base, f"lowlight_l{i}")
        os.makedirs(save_dir, exist_ok=True)
        for img_path in tqdm(img_paths, desc=f"lowlight_l{i}"):
            img = cv2.imread(img_path)
            if img is None:
                continue
            out = simulate_lowlight(
                img,
                gamma=setting["gamma"],
                alpha=setting["alpha"],
                beta=setting["beta"],
                alpha_s=setting["alpha_s"],
                sigma=setting["sigma"]
            )
            save_path = os.path.join(save_dir, os.path.basename(img_path))
            cv2.imwrite(save_path, out)


# ============================================================
# ---------- Entry point ----------
# ============================================================
if __name__ == "__main__":
    process_occlusion_dataset()
    process_lowlight_dataset()
    print("\n✅ All augmentations finished.")
    print("Output structure example:")
    print("   occ_l1/  → occlusion ratio 0.1")
    print("   occ_l2/  → occlusion ratio 0.2")
    print("   occ_l3/  → occlusion ratio 0.3")
    print("   lowlight_l1/ → low-light_level1")
    print("   lowlight_l2/ → low-light_level2")
    print("   lowlight_l3/ → low-light_level3")
