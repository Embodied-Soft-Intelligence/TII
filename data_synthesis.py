import os
import cv2
import numpy as np
import random
from glob import glob
from tqdm import tqdm

# ============================================================
# Input & Output Paths
# ============================================================
input_root = "/data/VMRD/images"
output_lowlight = r"/data/VRMD_LO/low"
output_occlusion = r"/data/VRMD_LO/occ"

os.makedirs(output_lowlight, exist_ok=True)
os.makedirs(output_occlusion, exist_ok=True)

# ============================================================
# Utility Functions
# ============================================================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def copy_all_images(src_folder, dst_folder):
    ensure_dir(dst_folder)
    img_paths = glob(os.path.join(src_folder, "*.jpg")) + glob(os.path.join(src_folder, "*.png"))
    for p in img_paths:
        cv2.imwrite(os.path.join(dst_folder, os.path.basename(p)), cv2.imread(p))
    return img_paths

# ============================================================
# Lowlight Functions
# ============================================================
def gamma_correction(img, gamma):
    img_norm = img / 255.0
    img_gamma = np.power(img_norm, gamma)
    return np.uint8(img_gamma * 255)

def adjust_contrast_brightness(img, alpha, beta):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def adjust_saturation(img, alpha_s):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1] = np.clip(hsv[:,:,1]*alpha_s,0,255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def add_gaussian_noise(img, sigma):
    noise = np.random.normal(0,sigma,img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.uint8(np.clip(noisy,0,255))

def random_lowlight(img):
    gamma = random.uniform(0.3,0.8)
    alpha = random.uniform(0.4,0.8)
    beta = random.uniform(-60,-10)
    alpha_s = random.uniform(0.3,0.9)
    sigma = random.uniform(3,20)
    out = gamma_correction(img,gamma)
    out = adjust_contrast_brightness(out,alpha,beta)
    out = adjust_saturation(out,alpha_s)
    out = add_gaussian_noise(out,sigma)
    return out

def gradient_lowlight(img, level_idx, num_levels):
    s = level_idx/(num_levels-1)
    gamma = 0.8 - s*(0.8-0.3)
    alpha = 0.8 - s*(0.8-0.4)
    beta = -10 + s*(-60+10)
    alpha_s = 0.9 - s*(0.9-0.3)
    sigma = 3 + s*(20-3)
    out = gamma_correction(img,gamma)
    out = adjust_contrast_brightness(out,alpha,beta)
    out = adjust_saturation(out,alpha_s)
    out = add_gaussian_noise(out,sigma)
    return out

# ============================================================
# Occlusion Functions
# ============================================================
def rotate_block(block,angle):
    h,w = block.shape[:2]
    M = cv2.getRotationMatrix2D((w/2,h/2),angle,1.0)
    cos = abs(M[0,0])
    sin = abs(M[0,1])
    new_w = int(h*sin + w*cos)
    new_h = int(h*cos + w*sin)
    M[0,2] += (new_w-w)/2
    M[1,2] += (new_h-h)/2
    rotated = cv2.warpAffine(block,M,(new_w,new_h),flags=cv2.INTER_LINEAR,borderValue=(0,0,0))
    mask = np.ones((h,w),dtype=np.uint8)*255
    rotated_mask = cv2.warpAffine(mask,M,(new_w,new_h),flags=cv2.INTER_NEAREST,borderValue=0)
    return rotated,rotated_mask

def random_occlusion(img):
    h_img,w_img = img.shape[:2]
    ratio = random.uniform(0.05,0.4)
    occ_h = max(1,int(h_img*ratio))
    occ_w = max(1,int(w_img*ratio))
    block = np.zeros((occ_h,occ_w,3),dtype=np.uint8)
    angle = random.uniform(-45,45)
    rot_block,rot_mask = rotate_block(block,angle)
    alpha = random.uniform(0.25,1.0)
    rh,rw = rot_block.shape[:2]
    if rh>h_img or rw>w_img:
        scale = min(h_img/rh,w_img/rw)*0.9
        rot_block = cv2.resize(rot_block,(int(rw*scale),int(rh*scale)))
        rot_mask = cv2.resize(rot_mask,(int(rw*scale),int(rh*scale)))
        rh,rw = rot_block.shape[:2]
    x1 = random.randint(0,w_img-rw)
    y1 = random.randint(0,h_img-rh)
    x2,y2 = x1+rw,y1+rh
    roi = img[y1:y2,x1:x2].astype(np.float32)
    fb = rot_block.astype(np.float32)
    mask_bool = (rot_mask>0).astype(np.float32)[...,None]
    blended = (alpha*fb + (1-alpha)*roi)*mask_bool + roi*(1-mask_bool)
    blended = blended.astype(np.uint8)
    out = img.copy()
    out[y1:y2,x1:x2] = blended
    return out

def gradient_occlusion(img, level_idx, num_levels, max_ratio=0.4):
    h_img,w_img = img.shape[:2]
    ratio = (level_idx/(num_levels-1))*max_ratio
    occ_h = max(1,int(h_img*ratio))
    occ_w = max(1,int(w_img*ratio))
    block = np.zeros((occ_h,occ_w,3),dtype=np.uint8)
    angle = random.uniform(-45,45)
    alpha = random.uniform(0.25,1.0)
    rot_block,rot_mask = rotate_block(block,angle)
    rh,rw = rot_block.shape[:2]
    if rh>h_img or rw>w_img:
        scale = min(h_img/rh,w_img/rw)*0.9
        rot_block = cv2.resize(rot_block,(int(rw*scale),int(rh*scale)))
        rot_mask = cv2.resize(rot_mask,(int(rw*scale),int(rh*scale)))
        rh,rw = rot_block.shape[:2]
    x1 = random.randint(0,w_img-rw)
    y1 = random.randint(0,h_img-rh)
    x2,y2 = x1+rw,y1+rh
    roi = img[y1:y2,x1:x2].astype(np.float32)
    fb = rot_block.astype(np.float32)
    mask_bool = (rot_mask>0).astype(np.float32)[...,None]
    blended = (alpha*fb + (1-alpha)*roi)*mask_bool + roi*(1-mask_bool)
    blended = blended.astype(np.uint8)
    out = img.copy()
    out[y1:y2,x1:x2] = blended
    return out

# ============================================================
# Processing Functions
# ============================================================
def process_train(split):
    print(f"\n--- Processing TRAIN {split} ---")
    src_folder = os.path.join(input_root, split)
    low_dst = os.path.join(output_lowlight, split)
    occ_dst = os.path.join(output_occlusion, split)
    img_paths = copy_all_images(src_folder, low_dst)
    copy_all_images(src_folder, occ_dst)
    total = len(img_paths)
    k = total//4
    selected = random.sample(img_paths,k)
    print(f"{split} train: augmenting {k} images")
    for path in tqdm(selected, desc="Lowlight train"):
        img = cv2.imread(path)
        out = random_lowlight(img)
        save_path = os.path.join(low_dst, os.path.basename(path))
        cv2.imwrite(save_path, out)
    for path in tqdm(selected, desc="Occlusion train"):
        img = cv2.imread(path)
        out = random_occlusion(img)
        save_path = os.path.join(occ_dst, os.path.basename(path))
        cv2.imwrite(save_path, out)

def process_val(split, num_levels=20):
    print(f"\n--- Processing VAL {split} ---")
    src_folder = os.path.join(input_root, split)
    img_paths = glob(os.path.join(src_folder,"*.jpg")) + glob(os.path.join(src_folder,"*.png"))
    # Lowlight gradient
    for level in range(num_levels):
        dst_folder = os.path.join(output_lowlight, split,f"level_{level+1:02d}")
        ensure_dir(dst_folder)
        for path in tqdm(img_paths, desc=f"Lowlight level {level+1}"):
            img = cv2.imread(path)
            out = gradient_lowlight(img, level, num_levels)
            save_path = os.path.join(dst_folder, os.path.basename(path))
            cv2.imwrite(save_path, out)
    # Occlusion gradient
    for level in range(num_levels):
        dst_folder = os.path.join(output_occlusion, split,f"level_{level+1:02d}")
        ensure_dir(dst_folder)
        for path in tqdm(img_paths, desc=f"Occlusion level {level+1}"):
            img = cv2.imread(path)
            out = gradient_occlusion(img, level, num_levels)
            save_path = os.path.join(dst_folder, os.path.basename(path))
            cv2.imwrite(save_path, out)

# ============================================================
# Main
# ============================================================
if __name__=="__main__":
    process_train("train")
    process_val("val", num_levels=20)
    print("\n✅ All processing finished.")
