# Face Classification

A tool for analyzing facial features using AI vision models to match them with feature descriptions.

## Overview

This repository contains tools to analyze facial images and classify them according to facial feature descriptions. It uses Azure AI Vision models to perform the analysis.

## Requirements

- Python 3.12+
- Azure AI Vision API credentials

## Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:Terra-Technologies/Vision-Modell-Demo.git
   cd Vision-Modell-Demo
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the root directory with the following variables:
   ```
   AZURE_ENDPOINT=your_azure_endpoint
   AZURE_API_KEY=your_azure_api_key
   AZURE_MODEL_NAME=your_azure_model_name
   ```

## Usage

### Analyzing Facial Features

To analyze a facial image against a set of descriptions:

```bash
python -m pretrained_vision_model --descriptions path/to/descriptions.txt --image path/to/image.jpg
```

Where:
- `path/to/descriptions.txt` is a text file containing facial feature descriptions, one per line
- `path/to/image.jpg` is the facial image you want to analyze

The tool will output the description that best matches the facial features in the image.

### Creating Sample Images from Kaggle Dataset

This repository includes a utility to extract facial images from the Kaggle Facial Keypoints Detection dataset:

1. Download the dataset from Kaggle and place it in `data/kaggle/facial-keypoints-detection/`

2. Run the extraction script:
   ```bash
   python scripts/kaggle/facial-keypoints-detection/create_images.py --num_images 100 --keypoints
   ```

   Options:
   - `--num_images`: Number of images to extract
   - `--keypoints`: Include facial keypoints in the extracted images (optional)

   The images will be saved to `data/images/`.

## Example

1. Create a file `descriptions.txt` with Ayurvedic facial feature descriptions:
   ```
    Full Lips: Naturally plump and balanced, both the upper and lower lips are equally full, giving a lush and voluminous appearance. Often seen as a symbol of beauty and sensuality.
    Thin Lips: The upper lip is fairly thin, while the lower lip, too, is slim—giving a dainty, delicate look. Thin lips have a crisp sharp lip line common for those with its a defined lip line.
    Round Lips: These are soft, round-shaped lips that usually appear young and inviting. The midsection of the lips is usually the most plump.
    Heart-Shaped Lips: Having a pronounced Cupid's bow and a slightly fuller bottom lip, this lip shape gives off a romantic and unique appearance.
    Wide Lips: Often, these lips spread wider over the face, which creates quite a vigorous, significant look. They can be full, or they can be thin.
    Bow-Shaped Lips: This classic shape features a clearly defined Cupid's bow with symmetry and fullness on both sides and is often considered to be elegant and well balanced.
   ```

2. Run the analysis:
   ```bash
   python -m pretrained_vision_model --descriptions descriptions.txt --image data/images/image_1.png
   ```