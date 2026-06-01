#!/usr/bin/env python3
"""
Visualize COCO segmentation labels with Vietnamese class names.

This script:
1. Reads annotations from a COCO JSON file
2. Finds corresponding images in the images directory
3. Visualizes segmentation masks with Vietnamese class names
4. Supports interactive viewing or batch saving
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path
import random
from PIL import Image, ImageDraw, ImageFont
import os

def generate_colors(num_classes):
    """Generate distinct colors for each class"""
    colors = []
    for i in range(num_classes):
        # Generate vibrant colors
        hue = i * 137.508  # Use golden angle for better distribution
        color = cv2.cvtColor(np.uint8([[[hue % 180, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(map(int, color)))
    return colors

def draw_vietnamese_text(img, text, position, font_path=None, font_size=20, color=(255, 255, 255)):
    """Draw Vietnamese text on image using PIL"""
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    
    # Try to use a font that supports Vietnamese
    try:
        if font_path and Path(font_path).exists():
            font = ImageFont.truetype(font_path, font_size)
        else:
            # Try common Vietnamese fonts
            for font_name in [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
                "arial.ttf"
            ]:
                if Path(font_name).exists():
                    font = ImageFont.truetype(font_name, font_size)
                    break
            else:
                font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # Draw text with background for better visibility
    bbox = draw.textbbox(position, text, font=font)
    draw.rectangle(bbox, fill=(0, 0, 0, 180))
    draw.text(position, text, font=font, fill=color)
    
    # Convert back to BGR
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img_bgr

def visualize_image(image_path, annotations, class_names, colors, font_path=None, alpha=0.4):
    """Visualize a single image with its COCO labels"""
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None
    
    # Create overlay for masks
    overlay = img.copy()
    
    # Draw each annotation
    for ann in annotations:
        class_id = ann['category_id']
        segmentation = ann.get('segmentation', [])
        
        if class_id not in class_names:
            continue
        
        class_name = class_names[class_id]
        
        # Color indexing: map class_id to a color deterministically
        # In COCO, category ids might not be contiguous and start from 1
        # We can just hash the id or map it to 0..N
        color = colors[hash(class_id) % len(colors)]
        
        if isinstance(segmentation, list):
            for seg in segmentation:
                if len(seg) < 6:
                    continue # Need at least 3 points
                points = np.array(seg, dtype=np.int32).reshape(-1, 2)
                
                # Draw filled polygon (mask)
                cv2.fillPoly(overlay, [points], color)
                
                # Draw polygon outline
                cv2.polylines(img, [points], isClosed=True, color=color, thickness=2)
                
                # Calculate centroid for label placement
                M = cv2.moments(points)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = points[0]
                
                # Draw class name (convert BGR to RGB for PIL)
                text_color = (color[2], color[1], color[0])  # BGR to RGB
                img = draw_vietnamese_text(img, class_name, (cx, cy), font_path=font_path, 
                                           font_size=20, color=text_color)
    
    # Blend overlay with original image
    result = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Visualize COCO segmentation labels with Vietnamese class names")
    parser.add_argument(
        '--images-dir',
        type=Path,
        required=True,
        help='Directory containing images'
    )
    parser.add_argument(
        '--coco-json',
        type=Path,
        required=True,
        help='Path to COCO instances JSON file'
    )
    parser.add_argument(
        '--image-path',
        type=Path,
        nargs='+',
        help='Path to one or more specific image files'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Directory to save visualized images (if not specified, shows interactively)'
    )
    parser.add_argument(
        '--font-path',
        type=Path,
        help='Path to TrueType font file for Vietnamese text'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.4,
        help='Transparency of mask overlay (0.0 to 1.0, default: 0.4)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of images to process'
    )
    args = parser.parse_args()

    # Load COCO JSON
    print(f"Loading COCO annotations from {args.coco_json}...")
    with open(args.coco_json, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # Build category map
    class_names = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
    print(f"Loaded {len(class_names)} classes.")
    
    # Generate colors
    # We will use length of class_names, max 100 as fallback
    num_colors = max(len(class_names), 100)
    colors = generate_colors(num_colors)
    
    # Build image map
    image_id_to_file = {img['id']: img['file_name'] for img in coco_data.get('images', [])}
    file_to_image_id = {img['file_name']: img['id'] for img in coco_data.get('images', [])}
    
    # Build annotation map
    image_id_to_anns = {}
    for ann in coco_data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in image_id_to_anns:
            image_id_to_anns[img_id] = []
        image_id_to_anns[img_id].append(ann)
    
    # Get images to process
    if args.image_path:
        image_files = []
        for p in args.image_path:
            if p.is_dir():
                found = list(p.glob("*.jpg")) + list(p.glob("*.jpeg")) + list(p.glob("*.png"))
                random.shuffle(found)
                image_files.extend(found)
            else:
                image_files.append(p)
    else:
        # Instead of globbing, we can just use the images defined in the JSON file
        # or list files in the directory
        image_files = (list(args.images_dir.glob("*.jpg")) + 
                      list(args.images_dir.glob("*.jpeg")) + 
                      list(args.images_dir.glob("*.png")))
        random.shuffle(image_files)

    # Always apply limit if specified
    if args.limit and len(image_files) > args.limit:
        image_files = image_files[:args.limit]
    
    print(f"Found {len(image_files)} images to process")
    
    # Create output directory if saving
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving visualizations to {args.output_dir}")
    
    # Process images
    total_processed = 0
    for i, image_path in enumerate(image_files, 1):
        # The file_name in COCO could be just the filename, or a relative path
        # Try both the stem + suffix and just the name
        file_name = image_path.name
        img_id = file_to_image_id.get(file_name)
        if img_id is None:
            # Fallback for checking if JSON has path e.g. "train/image.jpg"
            for k, v in file_to_image_id.items():
                if k.endswith(file_name):
                    img_id = v
                    break
        
        if img_id is None:
            print(f"[{i}/{len(image_files)}] Skipping {file_name} (Not found in COCO JSON)")
            continue
            
        annotations = image_id_to_anns.get(img_id, [])
        
        print(f"[{i}/{len(image_files)}] Processing {file_name} with {len(annotations)} annotations...", end=' ')
        
        result = visualize_image(image_path, annotations, class_names, colors,
                                 args.font_path, args.alpha)

        if result is None:
            print("FAILED")
            continue

        if args.output_dir:
            output_path = args.output_dir / image_path.name
            cv2.imwrite(str(output_path), result)
            print("SAVED")
            total_processed += 1
        else:
            print("SHOWING (press any key to continue, 'q' to quit)")
            cv2.imshow(f'Visualization - {image_path.name}', result)
            key = cv2.waitKey(0)
            if key == ord('q'):
                break
            cv2.destroyAllWindows()
            total_processed += 1

    if not args.output_dir:
        cv2.destroyAllWindows()

    print(f"\n✅ Processed {total_processed} images")

if __name__ == '__main__':
    main()
