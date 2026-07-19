import os

import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import InterpolationMode, Resize

if not os.path.exists("data"):
	os.makedirs("data")

# Resize images to 224x224 and save to data/
for folder in os.listdir("raw_data"):
	if not os.path.exists(f"data/{folder}"):
		os.makedirs(f"data/{folder}")
	for img in os.listdir(f"raw_data/{folder}"):
		if img.endswith(".png") or img.endswith(".jpg"):
			try:
				image = Image.open(f"raw_data/{folder}/{img}")
				transform = Resize((224, 224), interpolation=InterpolationMode.NEAREST)
				image = transform(image)
				image.save(f"data/{folder}/{img}")
			except Exception as e:
				print(f"Error processing {img} in {folder}: {e}")

# For each resized image, create a silhouette version and save it
for folder in os.listdir("data"):
	for img_name in os.listdir(f"data/{folder}"):
		img = cv2.imread(f"data/{folder}/{img_name}", cv2.IMREAD_UNCHANGED)
		if img is None:
			continue
		h, w = img.shape[:2]
		if len(img.shape) < 3:  # Grayscale image
			_, mask = cv2.threshold(
				img, 254, 255, cv2.THRESH_BINARY_INV
			)  # Assuming white background, convert everything else to black
			mask = np.vstack(
				(np.zeros((1, w), dtype="uint8"), mask, np.zeros((1, w), dtype="uint8"))
			)  # Add border to avoid edge touching
			mask = np.hstack(
				(
					np.zeros((h + 2, 1), dtype="uint8"),
					mask,
					np.zeros((h + 2, 1), dtype="uint8"),
				)
			)  # Add border to avoid edge touching
			cv2.floodFill(
				mask, None, (0, 0), 100
			)  # Flood fill from point (0,0) (changing background to gray)
			mask = np.where((mask == 100), 255, 0).astype(
				"uint8"
			)  # Change background to white, everything else to black
			mask = mask[1:-1, 1:-1]  # Remove border
		elif img.shape[2] == 3:  # Colored without alpha channel
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			_, mask = cv2.threshold(
				gray, 254, 255, cv2.THRESH_BINARY_INV
			)  # Assuming white background, convert everything else to black
			mask = np.vstack(
				(np.zeros((1, w), dtype="uint8"), mask, np.zeros((1, w), dtype="uint8"))
			)  # Add border to avoid edge touching
			mask = np.hstack(
				(
					np.zeros((h + 2, 1), dtype="uint8"),
					mask,
					np.zeros((h + 2, 1), dtype="uint8"),
				)
			)  # Add border to avoid edge touching
			cv2.floodFill(
				mask, None, (0, 0), 100
			)  # Flood fill from point (0,0) (changing background to gray)
			mask = np.where((mask == 100), 255, 0).astype(
				"uint8"
			)  # Change background to white, everything else to black
			mask = mask[1:-1, 1:-1]  # Remove border
		else:  # Colored with alpha channel
			mask = img[:, :, 3]  # Use alpha channel as mask
			mask = cv2.bitwise_not(mask)  # Invert mask (assuming transparent background)
		cv2.imwrite(f"data/{folder}/{img_name}", mask)
