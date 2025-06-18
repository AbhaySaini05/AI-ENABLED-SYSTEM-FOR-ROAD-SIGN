import argparse
import os
from pathlib import Path
from PIL import Image
import shutil

def convert_gt_to_yolo(gt_path, images_dir, labels_dir):
    """Convert ground truth file to YOLO format labels"""
    annotations = {}
    
    with open(gt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(';')
            if len(parts) != 6:
                continue
                
            img_name = parts[0]
            left, top, right, bottom = map(int, parts[1:5])
            class_id = int(parts[5])
            
            if img_name not in annotations:
                annotations[img_name] = []
            annotations[img_name].append((left, top, right, bottom, class_id))

    # Process all annotations
    for img_name, boxes in annotations.items():
        img_path = Path(images_dir) / img_name
        if not img_path.exists():
            continue

        # Get image dimensions
        with Image.open(img_path) as img:
            img_w, img_h = img.size

        # Create label file
        label_path = Path(labels_dir) / f"{Path(img_name).stem}.txt"
        with open(label_path, 'w') as f:
            for box in boxes:
                left, top, right, bottom, class_id = box
                
                # Convert to YOLO format
                x_center = (left + right) / 2 / img_w
                y_center = (top + bottom) / 2 / img_h
                width = (right - left) / img_w
                height = (bottom - top) / img_h
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def process_dataset(input_dir, output_images, output_labels, convert_labels=True):
    """Process a dataset split"""
    Path(output_images).mkdir(parents=True, exist_ok=True)
    Path(output_labels).mkdir(parents=True, exist_ok=True)

    # Convert PPM to JPG
    for ppm_path in Path(input_dir).glob('*.ppm'):
        jpg_path = Path(output_images) / f"{ppm_path.stem}.jpg"
        Image.open(ppm_path).save(jpg_path)
        
    # Convert labels if requested and gt.txt exists
    if convert_labels:
        gt_path = Path(input_dir) / 'gt.txt'
        if gt_path.exists():
            convert_gt_to_yolo(gt_path, input_dir, output_labels)
        else:
            print(f"No labels found in {input_dir}, skipping label conversion")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert GTSDB to YOLOv5 format')
    parser.add_argument('--train_dir', type=str, required=True, help='Training folder with PPMs and gt.txt')
    parser.add_argument('--test_dir', type=str, required=True, help='Test folder with PPMs only')
    parser.add_argument('--output_dir', type=str, default='yolov5_data', help='Output directory')
    
    args = parser.parse_args()

    # Process training data with labels
    process_dataset(
        input_dir=args.train_dir,
        output_images=Path(args.output_dir) / 'images/train',
        output_labels=Path(args.output_dir) / 'labels/train',
        convert_labels=True
    )

    # Process test data without labels
    process_dataset(
        input_dir=args.test_dir,
        output_images=Path(args.output_dir) / 'images/test',
        output_labels=Path(args.output_dir) / 'labels/test',
        convert_labels=False
    )

    # Create dataset.yaml
    yaml_content = f"""path: {Path(args.output_dir).absolute()}
train: images/train
val: images/test  # For inference only - not actual validation
nc: 43  # UPDATE WITH ACTUAL CLASS COUNT
names: []  # ADD CLASS NAMES
"""

    with open(Path(args.output_dir) / 'dataset.yaml', 'w') as f:
        f.write(yaml_content)

    print(f'Conversion complete. Output in {args.output_dir}')