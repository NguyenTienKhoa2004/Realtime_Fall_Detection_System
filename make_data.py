import os
import cv2
import mediapipe as mp
import csv
import numpy as np

def process_dataset(input_folder, output_csv, max_images=None):
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    # static_image_mode=True is optimized for processing independent images
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3)

    # Check if data folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Folder '{input_folder}' does not exist.")
        return

    # Prepare CSV header
    # We have 33 landmarks. Each has x, y, z, visibility.
    # Structure: class, x1, y1, z1, v1, x2, y2, z2, v2, ...
    header = ['class']
    for i in range(1, 34): # 1 to 33
        header += [f'x{i}', f'y{i}', f'z{i}', f'v{i}']

    print(f"Processing {input_folder} -> {output_csv}...")
    
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        # Iterate through each class folder in data_da_chia
        class_names = os.listdir(input_folder)
        
        for class_name in class_names:
            class_path = os.path.join(input_folder, class_name)
            
            if not os.path.isdir(class_path):
                continue

            print(f"Processing class: {class_name}...")
            
            image_files = os.listdir(class_path)
            np.random.shuffle(image_files) # Shuffle to ensure random selection if limiting
            count = 0
            
            for img_name in image_files:
                if max_images is not None and count >= max_images:
                    break

                img_path = os.path.join(class_path, img_name)
                
                # Read image
                image = cv2.imread(img_path)
                if image is None:
                    continue

                # Convert to RGB for MediaPipe
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe
                results = pose.process(image_rgb)
                
                if results.pose_landmarks:
                    # Extract coordinates and flatten to [x1, y1, z1, v1, x2, ...]
                    # This matches the logic in client.py
                    pose_row = list(np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten())
                    
                    # Write row: [class_name, coords...]
                    row = [class_name] + pose_row
                    writer.writerow(row)
                    count += 1
                else:
                    print(f"  [Skipped] No pose detected: {img_name}")
            
            print(f"  -> Extracted features from {count} images for '{class_name}'")

    print(f"Done! Data saved to '{output_csv}'")

if __name__ == "__main__":
    # Create Train CSV
    process_dataset('dataset_da_chia/train', 'train.csv', max_images=120)
    
    # Create Test CSV
    process_dataset('dataset_da_chia/test', 'test.csv')