#!/usr/bin/env python3
"""
Visualize YOLO segmentation labels with Vietnamese class names.

This script:
1. Reads images from an images directory
2. Reads corresponding YOLO labels from a labels directory
3. Visualizes segmentation masks with Vietnamese class names
4. Supports interactive viewing or batch saving
"""

import cv2
import numpy as np
import yaml
import argparse
from pathlib import Path
import random
from PIL import Image, ImageDraw, ImageFont

def load_class_names(data_yaml_path):
    """Load class names from data.yaml"""
    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data['names']

def generate_colors(num_classes):
    """Generate distinct colors for each class"""
    random.seed(42)
    colors = []
    for i in range(num_classes):
        # Generate vibrant colors
        hue = i * 137.508  # Use golden angle for better distribution
        color = cv2.cvtColor(np.uint8([[[hue % 180, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(map(int, color)))
    return colors

def parse_yolo_label(label_path, img_width, img_height):
    """Parse YOLO segmentation label file"""
    annotations = []
    
    if not label_path.exists():
        return annotations
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            class_id = int(parts[0])
            # Parse polygon points (normalized coordinates)
            coords = list(map(float, parts[1:]))
            
            # Convert to absolute coordinates
            points = []
            for i in range(0, len(coords), 2):
                x = int(coords[i] * img_width)
                y = int(coords[i + 1] * img_height)
                points.append([x, y])
            
            annotations.append({
                'class_id': class_id,
                'points': np.array(points, dtype=np.int32)
            })
    
    return annotations

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

def visualize_image(image_path, label_path, class_names, colors, font_path=None, alpha=0.4):
    """Visualize a single image with its YOLO labels"""
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None
    
    h, w = img.shape[:2]
    
    # Parse labels
    annotations = parse_yolo_label(label_path, w, h)
    
    # Create overlay for masks
    overlay = img.copy()
    
    # Draw each annotation
    for ann in annotations:
        class_id = ann['class_id']
        points = ann['points']
        
        if class_id not in class_names:
            continue
        
        class_name = class_names[class_id]
        color = colors[class_id % len(colors)]
        
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
    parser = argparse.ArgumentParser(description="Visualize YOLO segmentation labels with Vietnamese class names")
    parser.add_argument(
        '--images-dir',
        type=Path,
        required=True,
        help='Directory containing images'
    )
    parser.add_argument(
        '--labels-dir',
        type=Path,
        required=True,
        help='Directory containing YOLO label files'
    )
    parser.add_argument(
        '--data-yaml',
        type=Path,
        required=True,
        help='Path to data.yaml containing class names'
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
    
    # Load class names
    class_names = load_class_names(args.data_yaml)
    print(f"Loaded {len(class_names)} classes from {args.data_yaml}")
    
    # Generate colors
    colors = generate_colors(len(class_names))
    
    # Get all images
    image_files = sorted(list(args.images_dir.glob("*.jpg")) + 
                        list(args.images_dir.glob("*.jpeg")) + 
                        list(args.images_dir.glob("*.png")))
    
    if args.limit:
        image_files = image_files[:args.limit]
    
    print(f"Found {len(image_files)} images")
    
    # Create output directory if saving
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving visualizations to {args.output_dir}")
    
    # Process each image
    for i, image_path in enumerate(image_files, 1):
        label_path = args.labels_dir / (image_path.stem + '.txt')
        
        print(f"[{i}/{len(image_files)}] Processing {image_path.name}...", end=' ')
        
        # Visualize
        result = visualize_image(image_path, label_path, class_names, colors, 
                                args.font_path, args.alpha)
        
        if result is None:
            print("FAILED")
            continue
        
        if args.output_dir:
            # Save to output directory
            output_path = args.output_dir / image_path.name
            cv2.imwrite(str(output_path), result)
            print("SAVED")
        else:
            # Show interactively
            print("SHOWING (press any key to continue, 'q' to quit)")
            cv2.imshow(f'Visualization - {image_path.name}', result)
            key = cv2.waitKey(0)
            if key == ord('q'):
                break
            cv2.destroyAllWindows()
    
    if not args.output_dir:
        cv2.destroyAllWindows()
    
    print(f"\n✅ Processed {len(image_files)} images")

if __name__ == '__main__':
    main()
