import json
import os
from PIL import Image

def get_image_dimensions(img_path):
    """Get image width and height."""
    try:
        img = Image.open(img_path)
        return img.width, img.height
    except Exception as e:
        print(f"Warning: Could not read image {img_path}: {e}")
        return None, None

def convert(json_path, output_dir, img_width, img_height):
    """Convert a single JSON label file to YOLO format."""
    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return

    txt_name = os.path.basename(json_path).replace('.json', '.txt')
    out_path = os.path.join(output_dir, txt_name)

    with open(out_path, 'w') as out:
        for shape in data['shapes']:
            label = shape['label']
            points = shape['points']

            x1, y1 = points[0]
            x2, y2 = points[1]

            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            w = abs(x2 - x1) / img_width
            h = abs(y2 - y1) / img_height

            # Clamp normalized coordinates to [0, 1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            w = max(0, min(1, w))
            h = max(0, min(1, h))

            class_id = int(label)  # Use the label as class_id

            out.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

def convert_all_labels(labels_dir, output_dir, images_dir):
    """Convert all JSON labels in a directory to YOLO format."""
    os.makedirs(output_dir, exist_ok=True)
    
    json_files = [f for f in os.listdir(labels_dir) if f.endswith('.json')]
    
    print(f"Found {len(json_files)} JSON files to convert")
    
    for json_file in json_files:
        json_path = os.path.join(labels_dir, json_file)
        
        # Try to find corresponding image
        base_name = json_file.replace('.json', '')
        img_path = None
        
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            candidate = os.path.join(images_dir, base_name + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        
        if img_path:
            img_width, img_height = get_image_dimensions(img_path)
            if img_width and img_height:
                convert(json_path, output_dir, img_width, img_height)
                print(f"Converted: {json_file}")
            else:
                print(f"Skipped: {json_file} (could not read image dimensions)")
        else:
            print(f"Skipped: {json_file} (no corresponding image found)")

if __name__ == '__main__':
    # Configuration
    labels_dir = './data/labels/val'
    output_dir = './data/labels/val'
    images_dir = './data/images/val'
    
    convert_all_labels(labels_dir, output_dir, images_dir)
    print("Conversion complete!")
