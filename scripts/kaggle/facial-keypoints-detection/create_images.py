import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path

def load_and_save_images(num_images, include_keypoints=False):
    # Create output directory if it doesn't exist
    output_dir = Path('data/images')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the dataset
    try:
        data = pd.read_csv('data/kaggle/facial-keypoints-detection/training.csv')
        print(f"Loaded dataset with {len(data)} entries")
    except FileNotFoundError:
        print("Error: Dataset file not found. Make sure the path is correct.")
        return
    
    # Limit to the requested number of images
    if num_images > len(data):
        print(f"Warning: Requested {num_images} images but dataset only has {len(data)}. Using all available images.")
        num_images = len(data)
    
    data = data.iloc[:num_images]
    
    # Process and save each image
    for i, row in data.iterrows():
        # Convert the image string to a numpy array
        image = np.fromstring(row['Image'], sep=' ').reshape(96, 96)
        
        # Create a figure
        plt.figure(figsize=(5, 5))
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        
        # If keypoints are requested, plot them
        if include_keypoints:
            keypoints = row.drop('Image').values.reshape(-1, 2)
            # Filter out NaN values
            valid_keypoints = ~np.isnan(keypoints).any(axis=1)
            keypoints = keypoints[valid_keypoints]
            
            if len(keypoints) > 0:
                plt.scatter(keypoints[:, 0], keypoints[:, 1], s=20, marker='.', c='r')
        
        # Save the image
        plt.savefig(f"{output_dir}/image_{i+1}.png", bbox_inches='tight', pad_inches=0)
        plt.close()
        
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{num_images} images")
    
    print(f"Successfully saved {num_images} images to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract facial images from Kaggle dataset')
    parser.add_argument('--num_images', type=int, help='Number of images to extract')
    parser.add_argument('--keypoints', action='store_true', help='Include facial keypoints in the images')
    
    args = parser.parse_args()
    
    load_and_save_images(args.num_images, args.keypoints)