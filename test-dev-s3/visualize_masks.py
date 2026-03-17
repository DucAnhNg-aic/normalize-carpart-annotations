#!/usr/bin/env python3
"""
Script to download masks from CloudFront URLs and visualize them on the original image.
Modified to load from S3/CloudFront instead of MinIO.
"""

import json
import os
from pathlib import Path
import cv2
import numpy as np
import requests
from io import BytesIO

# CloudFront Configuration
CLOUDFRONT_BASE_URL = "https://dyta7vmv7sqle.cloudfront.net"

# Paths
RESULT_JSON = "res.json"
OUTPUT_DIR = "visualization"
TEMP_DIR = "/tmp/mask_download_s3"

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)


def download_from_url(url, local_path):
    """Download file from URL"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            f.write(response.content)
        
        print(f"✓ Downloaded: {url}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {url}: {e}")
        return False


def download_image_from_url(url):
    """Download image directly into memory and return as numpy array"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Convert to numpy array
        img_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            print(f"✗ Failed to decode image from {url}")
            return None
            
        print(f"✓ Downloaded image: {url}")
        return img
    except Exception as e:
        print(f"✗ Failed to download image {url}: {e}")
        return None


def generate_distinct_colors(n):
    """Generate n visually distinct colors using HSV color space"""
    colors = []
    for i in range(n):
        hue = int(180 * i / n)  # Spread across hue spectrum
        saturation = 255  # Full saturation for vibrant colors
        value = 200  # Bright but not too bright
        hsv_color = np.array([[[hue, saturation, value]]], dtype=np.uint8)
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(map(int, bgr_color)))
    return colors


def simplify_class_name(class_uuid):
    """
    Simplify Vietnamese class name by removing only the trailing UUID.
    
    Examples:
        'canh-cua-truoc-trai-9iR4GT' -> 'canh cua truoc trai'
        'noc-xe-tren-U5leeQ' -> 'noc xe tren'
    """
    # Split by hyphen
    parts = class_uuid.split('-')
    
    # Remove the last part (UUID - typically 6 characters alphanumeric)
    if len(parts) > 1 and len(parts[-1]) <= 8:
        parts = parts[:-1]
    
    # Join with spaces
    simplified_name = ' '.join(parts)
    
    return simplified_name


def visualize_result(result):
    """Visualize masks for a single result"""
    
    # Get image info
    img_url = result.get("imgUrl", "unknown.jpg")
    img_resize = result.get("imageResize", img_url)  # Use resized image if available
    img_resolution = result.get("imgResolution", [1066, 800]) # [W, H]
    
    print(f"\n{'='*60}")
    print(f"Processing: {img_url}")
    print(f"Using resized image: {img_resize}")
    print(f"{'='*60}")
    
    # Download resized image from CloudFront URL
    # Construct full URL: https://dyta7vmv7sqle.cloudfront.net/INSURANCE_RESULT/<filename>
    img_cloudfront_url = f"{CLOUDFRONT_BASE_URL}/{img_resize}"
    
    img = download_image_from_url(img_cloudfront_url)
    if img is None:
        print(f"✗ Failed to load image from {img_cloudfront_url}")
        return
    
    # Get actual image dimensions
    h, w = img.shape[:2]
    
    # API resolution should match the resized image dimensions
    api_w, api_h = img_resolution  # [W, H] from API
    
    print(f"Image size: {w}x{h}")
    print(f"API Resolution: {api_w}x{api_h}")
    
    # Check if dimensions match (they should if imageResize is correct)
    if w != api_w or h != api_h:
        print(f"⚠️  Warning: Image size mismatch! Using actual image size for coordinates.")
        print(f"   This may cause slight misalignment.")
    
    # Create overlay for masks
    overlay = img.copy()
    
    # Collect all unique class UUIDs and map to simplified names
    all_items = result.get("carParts", []) + result.get("carDamages", [])
    
    # Create mapping from UUID to simplified name
    uuid_to_simplified = {}
    for item in all_items:
        class_uuid = item.get("classUUID", "unknown")
        uuid_to_simplified[class_uuid] = simplify_class_name(class_uuid)
    
    # Get unique simplified names
    unique_simplified_names = list(set(uuid_to_simplified.values()))
    unique_simplified_names.sort()  # Sort for consistent color assignment
    
    # Generate distinct colors for each simplified name
    simplified_name_colors = {}
    distinct_colors = generate_distinct_colors(len(unique_simplified_names))
    for idx, simplified_name in enumerate(unique_simplified_names):
        simplified_name_colors[simplified_name] = distinct_colors[idx]
    
    # Create UUID to color mapping (via simplified name)
    class_colors = {}
    for class_uuid, simplified_name in uuid_to_simplified.items():
        class_colors[class_uuid] = simplified_name_colors[simplified_name]
    
    print(f"\nFound {len(unique_simplified_names)} unique simplified classes")
    for simplified_name, color in sorted(simplified_name_colors.items()):
        print(f"  {simplified_name[:30]:30s} -> RGB{color}")
    
    # Function to process and draw masks
    def process_items(items, is_damage=False):
        item_type = "damage" if is_damage else "car part"
        print(f"\nProcessing {len(items)} {item_type}s...")
        
        for idx, item in enumerate(items):
            class_uuid = item.get("classUUID", "unknown")
            mask_name = item.get("maskPath", "")
            score = item.get("score", 0.0)
            box = item.get("box", []) # [x1, y1, x2, y2] normalized
            
            if not mask_name or len(box) != 4:
                continue
            
            # Download mask from CloudFront URL
            mask_cloudfront_url = f"{CLOUDFRONT_BASE_URL}/INSURANCE_RESULT/{mask_name}"
            mask_local_path = os.path.join(TEMP_DIR, mask_name)
            
            if download_from_url(mask_cloudfront_url, mask_local_path):
                # Load cropped mask (with Alpha channel if exists)
                mask_crop_raw = cv2.imread(mask_local_path, cv2.IMREAD_UNCHANGED)
                if mask_crop_raw is None: continue

                # Extract proper mask channel
                if len(mask_crop_raw.shape) == 3 and mask_crop_raw.shape[2] == 4:
                    # Use Alpha channel as mask
                    mask_crop = mask_crop_raw[:, :, 3]
                else:
                    # Fallback for standard grayscale
                    mask_crop = cv2.imread(mask_local_path, cv2.IMREAD_GRAYSCALE)

                # Calculate absolute box coordinates [xmin, ymin, xmax, ymax]
                # box format from API: [x1, y1, x2, y2] (normalized 0-1)
                x1, y1, x2, y2 = (
                    int(box[0] * w),  # x1 * width
                    int(box[1] * h),  # y1 * height
                    int(box[2] * w),  # x2 * width
                    int(box[3] * h)   # y2 * height
                )
                
                box_w = max(1, x2 - x1)
                box_h = max(1, y2 - y1)
                
                # Resize cropped mask to fit the box using NEAREST neighbor to preserve shape
                mask_resized = cv2.resize(mask_crop, (box_w, box_h), interpolation=cv2.INTER_NEAREST)
                
                # Get color for this class (use red for damage, class color for parts)
                if is_damage:
                    color = (0, 0, 255)  # Red for damage
                else:
                    color = class_colors.get(class_uuid, (255, 255, 255))
                
                # Ensure we don't go out of bounds
                crop_h = min(box_h, h - y1)
                crop_w = min(box_w, w - x1)
                
                if crop_h <= 0 or crop_w <= 0: continue

                # Get the ROI on the main overlay
                roi = overlay[y1:y1+crop_h, x1:x1+crop_w]
                
                # Get the corresponding mask area
                mask_area = mask_resized[:crop_h, :crop_w]
                
                # Create a colored mask for the ROI
                colored_roi = roi.copy()
                
                # Apply color only to masked pixels
                mask_indices = mask_area > 0
                colored_roi[mask_indices] = color
                
                # Blend the colored ROI with the original ROI
                alpha = 0.5 # Transparency
                cv2.addWeighted(colored_roi, alpha, roi, 1 - alpha, 0, roi)
                
                # Replace the ROI in the overlay image
                overlay[y1:y1+crop_h, x1:x1+crop_w] = roi

                # Draw contours for sharper defined edges
                contours, _ = cv2.findContours(mask_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, color, 2, offset=(x1, y1))
                
                # --- Draw Label Logic: Place INSIDE or very close to mask ---
                # Use simplified class name (remove position and UUID)
                class_name_short = simplify_class_name(class_uuid)
                
                print(f"  [{idx+1}/{len(items)}] {class_name_short[:30]:30s} at [{x1},{y1},{x2},{y2}]")
                label_text = f"{class_name_short} {score:.2f}"

                
                font_scale = 0.3
                font_thickness = 1
                font = cv2.FONT_HERSHEY_SIMPLEX
                
                (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
                
                # Find the centroid of the mask to place label
                moments = cv2.moments(mask_area)
                if moments["m00"] != 0:
                    # Centroid relative to box
                    cx_rel = int(moments["m10"] / moments["m00"])
                    cy_rel = int(moments["m01"] / moments["m00"])
                    
                    # Absolute centroid
                    cx = x1 + cx_rel
                    cy = y1 + cy_rel
                    
                    # Position label at centroid (centered)
                    text_x = cx - text_w // 2
                    text_y = cy + text_h // 2
                else:
                    # Fallback: top-left of box
                    text_x = x1 + 5
                    text_y = y1 + text_h + 5
                
                # Ensure text stays within image bounds
                text_x = max(5, min(text_x, w - text_w - 5))
                text_y = max(text_h + 5, min(text_y, h - 5))
                
                # Draw background rectangle with class color border
                padding = 4
                bg_x1 = text_x - padding
                bg_y1 = text_y - text_h - padding
                bg_x2 = text_x + text_w + padding
                bg_y2 = text_y + padding
                
                # Ensure background stays in bounds
                bg_x1 = max(0, bg_x1)
                bg_y1 = max(0, bg_y1)
                bg_x2 = min(w, bg_x2)
                bg_y2 = min(h, bg_y2)
                
                # Draw semi-transparent white background
                sub_img = overlay[bg_y1:bg_y2, bg_x1:bg_x2]
                if sub_img.size > 0:
                    white_rect = np.full(sub_img.shape, 255, dtype=np.uint8)
                    res = cv2.addWeighted(sub_img, 0.3, white_rect, 0.7, 1.0)
                    overlay[bg_y1:bg_y2, bg_x1:bg_x2] = res
                    
                    # Draw colored border matching mask color (thicker for visibility)
                    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 3)

                # Draw text in black for contrast on white background
                cv2.putText(overlay, label_text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)

    # Process both types
    process_items(result.get("carParts", []), is_damage=False)
    process_items(result.get("carDamages", []), is_damage=True)
    
    # Save result
    output_filename = f"visualized_{os.path.basename(img_url)}"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    cv2.imwrite(output_path, overlay)
    print(f"\n✓ Saved visualization to: {output_path}")




def main():
    """Main function"""
    print("="*60)
    print("CloudFront Mask Visualization Script")
    print("="*60)
    
    # Load result JSON
    print(f"\nLoading results from: {RESULT_JSON}")
    with open(RESULT_JSON, 'r') as f:
        data = json.load(f)
    
    results = data.get("result", [])
    print(f"Found {len(results)} result(s)")
    
    # Process each result
    for idx, result in enumerate(results):
        print(f"\n{'#'*60}")
        print(f"Result {idx + 1}/{len(results)}")
        print(f"{'#'*60}")
        visualize_result(result)
    
    print("\n" + "="*60)
    print("✓ All visualizations completed!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
