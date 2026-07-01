import cv2
import numpy as np


def extract_features(image_path: str) -> dict:
    """
    Analyze one image and return numerical features.
    分析一张图片，返回数字特征。
    """

    img_bgr = cv2.imread(image_path)

    if img_bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    saturation = hsv[:, :, 1]
    total_pixels = gray.size

    brightness_mean = float(np.mean(gray))
    brightness_std = float(np.std(gray))

    shadow_clip_percent = float(np.sum(gray <= 3) / total_pixels * 100)
    highlight_clip_percent = float(np.sum(gray >= 252) / total_pixels * 100)

    saturation_mean = float(np.mean(saturation))
    saturation_std = float(np.std(saturation))

    red_mean = float(np.mean(img_rgb[:, :, 0]))
    green_mean = float(np.mean(img_rgb[:, :, 1]))
    blue_mean = float(np.mean(img_rgb[:, :, 2]))

    blue_red_ratio = blue_mean / max(red_mean, 1)
    blue_green_ratio = blue_mean / max(green_mean, 1)

    sharpness_laplacian = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    return {
        "brightness_mean": brightness_mean,
        "brightness_std": brightness_std,
        "contrast": brightness_std,
        "shadow_clip_percent": shadow_clip_percent,
        "highlight_clip_percent": highlight_clip_percent,
        "saturation_mean": saturation_mean,
        "saturation_std": saturation_std,
        "red_mean": red_mean,
        "green_mean": green_mean,
        "blue_mean": blue_mean,
        "blue_red_ratio": blue_red_ratio,
        "blue_green_ratio": blue_green_ratio,
        "sharpness_laplacian": sharpness_laplacian,
    }