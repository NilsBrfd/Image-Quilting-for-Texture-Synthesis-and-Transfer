from PIL import Image
import os
import numpy as np
from random import randint
import argparse


# ===================== ARGUMENTS =====================

def parse_args():
    parser = argparse.ArgumentParser(description="Random placement texture synthesis")

    parser.add_argument("--src", type=str, required=True, help="Source texture image")
    parser.add_argument("--patch", type=int, default=10, help="Number of patches per dimension")
    parser.add_argument("--multiply", type=float, default=2,help="Output scale factor")

    return parser.parse_args()


args = parse_args()

IMAGE_SRC = args.src
NOMBRE_CARRE = args.patch
MULTIPLY = args.multiply


# ===================== CHECK FILE =====================

if not os.path.exists(IMAGE_SRC):
    raise FileNotFoundError(f"Image not found: {IMAGE_SRC}")


# ===================== RANDOM PLACEMENT =====================

def get_random_placement(array, output, size_h, size_w):

    for i in range(0, output.shape[0], size_h):
        for j in range(0, output.shape[1], size_w):
            x = randint(0, array.shape[1] - size_w)
            y = randint(0, array.shape[0] - size_h)
            patch_h = min(size_h, output.shape[0] - i)
            patch_w = min(size_w, output.shape[1] - j)
            output[i:i + patch_h, j:j + patch_w] = array[
                y:y + patch_h,
                x:x + patch_w
            ]

    return output


# ===================== MAIN =====================

im = Image.open(IMAGE_SRC).convert("RGB")
arr = np.array(im)

output_size = (round(arr.shape[0] * MULTIPLY), round(arr.shape[1] * MULTIPLY), 3)

output = np.zeros(output_size, dtype=np.uint8)

block_h = output_size[0] // NOMBRE_CARRE
block_w = output_size[1] // NOMBRE_CARRE

print(f"Output size: {output_size}")
print(f"Patch size: {block_h}x{block_w}")

result = get_random_placement(arr, output, block_h,block_w)

base_name = os.path.splitext(os.path.basename(IMAGE_SRC))[0]
save_dir = f"results/random_placement/{base_name}_random_placement"

os.makedirs(save_dir, exist_ok=True)

filename = f"{base_name}_{NOMBRE_CARRE}_{MULTIPLY}.png"
path = os.path.join(save_dir, filename)

Image.fromarray(result.astype(np.uint8)).save(path)

print("Saved:", path)

Image.fromarray(result.astype(np.uint8)).show()