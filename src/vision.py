"""A deterministic colour-segmentation demonstration, not a trained disease classifier."""

import cv2
import numpy as np
from PIL import Image, ImageOps

SAMPLE_CROPS = {
    "healthy_leaf.png": "Unspecified",
    "tomato_early_blight.png": "Tomato",
    "potato_late_blight.png": "Potato",
    "corn_rust.png": "Maize",
    "rice_blast.png": "Rice",
}
ILLUSTRATIVE_MATCHES = {
    "Tomato": "Early-blight-like spotting",
    "Potato": "Late-blight-like dark patches",
    "Maize": "Rust-like coloured spots",
    "Rice": "Blast-like spotting",
}


def analyze_leaf(image, crop="Unspecified"):
    if crop not in ["Unspecified", *ILLUSTRATIVE_MATCHES]:
        raise ValueError("Unsupported crop hint.")
    if not isinstance(image, Image.Image):
        raise TypeError("Open the image using PIL.Image before analysis.")
    image = ImageOps.exif_transpose(image).convert("RGB")
    if min(image.size) < 16:
        raise ValueError("Use an image at least 16 by 16 pixels.")
    image.thumbnail((600, 600))
    rgb = np.asarray(image).copy()
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)
    green = (hue >= 32) & (hue <= 100) & (saturation > 40) & (value > 30)
    if green.sum() < 0.01 * green.size:
        raise ValueError("No clear green leaf region found. Use a leaf on a plain light background.")
    coloured = ((saturation > 35) & (value > 20)).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    coloured = cv2.morphologyEx(coloured, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(coloured, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Could not find a leaf boundary.")
    # Keep the largest region and fill internal holes so dark spots remain in the leaf.
    contour = max(contours, key=cv2.contourArea)
    leaf_mask = np.zeros(hue.shape, dtype=np.uint8)
    cv2.drawContours(leaf_mask, [contour], -1, 255, thickness=cv2.FILLED)
    leaf = leaf_mask > 0
    if leaf.sum() < 0.03 * leaf.size or (green & leaf).sum() < 0.01 * leaf.sum():
        raise ValueError("Leaf region is too small or unclear.")
    brown_or_rust = (hue < 27) & (saturation > 55) & (value > 25)
    yellow = (hue >= 20) & (hue < 36) & (saturation > 80) & (value > 100)
    dark = value < 75
    marked = (leaf & (brown_or_rust | yellow | dark)).astype(np.uint8) * 255
    marked = cv2.morphologyEx(marked, cv2.MORPH_OPEN, kernel)
    marked = cv2.bitwise_and(marked, leaf_mask)
    affected = marked > 0
    percent = float(affected.sum() / leaf.sum() * 100)
    band = "Low colour change" if percent < 2.5 else (
        "Mild" if percent < 10 else "Moderate" if percent <= 25 else "High"
    )
    overlay = rgb.copy()
    # Blend only marked pixels. Background and unaffected pixels remain unchanged.
    overlay[affected] = (0.5 * rgb[affected] + 0.5 * np.array([255, 35, 35])).astype(np.uint8)
    spot_contours, _ = cv2.findContours(marked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    match = "No substantial colour change detected"
    if percent >= 2.5:
        match = ILLUSTRATIVE_MATCHES.get(crop, "Colour changes detected; select a crop for an illustrative comparison")
    return {
        "original": rgb, "leaf_mask": leaf_mask, "affected_mask": marked, "overlay": overlay,
        "leaf_pixels": int(leaf.sum()), "affected_pixels": int(affected.sum()),
        "affected_percent": percent, "severity_band": band,
        "regions": len(spot_contours), "illustrative_match": match,
    }
