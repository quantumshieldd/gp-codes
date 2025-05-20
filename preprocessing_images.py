import os
import cv2
import numpy as np

image_dir = r'C:\Users\ASUS\Desktop\gp\training\image_2'
label_dir = r'C:\Users\ASUS\Desktop\gp\data_object_label_2\training\label_2'
output_dir = r'C:\Users\ASUS\Desktop\gp\processed_images'

os.makedirs(output_dir, exist_ok=True)

#resize all images to this
target_size = (512, 256) 

# Parse one line from KITTI label file
def parse_label_line(line, scale_x, scale_y):
    parts = line.strip().split()
    obj_class = parts[0]

    # Skip 'DontCare'labels
    if obj_class == 'DontCare':
        return None

    # Get 2D bounding box coordinates
    x1, y1, x2, y2 = map(float, parts[4:8])

    # Scale bbox to resized image dimensions
    x1 *= scale_x
    y1 *= scale_y
    x2 *= scale_x
    y2 *= scale_y

    return {
        'class': obj_class,
        'bbox': [x1, y1, x2, y2]
    }

#Process  images 
for filename in sorted(os.listdir(image_dir))[:]:  
    if not filename.endswith('.png'):
        continue

    image_id = filename.split('.')[0]
    image_path = os.path.join(image_dir, filename)
    label_path = os.path.join(label_dir, f'{image_id}.txt')

    # Load image
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    # Resize and normalize image
    image_resized = cv2.resize(image, target_size)
    image_resized = image_resized.astype(np.float32) / 255.0

    # Calculate scaling factors for resizing bboxes
    scale_x = target_size[0] / w
    scale_y = target_size[1] / h

    # Parse label file
    annotations = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parsed = parse_label_line(line, scale_x, scale_y)
                if parsed:
                    annotations.append(parsed)

    # Save image and annotations to .npy file
    output_path = os.path.join(output_dir, f'{image_id}.npy')
    np.save(output_path, {
        'image': image_resized,
        'annotations': annotations
    })

    print(f" Processed {filename} → {len(annotations)} annotation(s)")

print(" Preprocessing complete! All .npy files are saved.")
