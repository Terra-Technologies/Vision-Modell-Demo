import argparse
import os
import json
import base64
import requests
from dotenv import load_dotenv
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage, ImageContentItem, TextContentItem, ImageUrl
from azure.core.credentials import AzureKeyCredential

def read_descriptions(file_path):
    """Read facial feature descriptions from a text file."""
    with open(file_path, 'r') as f:
        descriptions = [line.strip() for line in f if line.strip()]
    return descriptions

def encode_image(image_path):
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_facial_features(image_path, descriptions):
    """Analyze facial features using Azure AI Inference."""
    load_dotenv()
    
    # Get Azure credentials from environment variables
    endpoint = os.getenv("AZURE_ENDPOINT")
    api_key = os.getenv("AZURE_API_KEY")
    model_name = os.getenv("AZURE_MODEL_NAME")
    
    if not all([endpoint, api_key, model_name]):
        raise ValueError("Azure credentials not found in environment variables")
    
    # Encode image
    base64_image = encode_image(image_path)
    
    # Format descriptions for the prompt
    descriptions_text = "\n".join([f"- {desc}" for desc in descriptions])
    
    # Initialize the client
    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(api_key),
    )
    
    # Create the request using the client
    response = client.complete(
        messages=[
            SystemMessage(content="You are a facial feature analysis expert. Look at the image and determine which of the provided descriptions best matches the facial features shown."),
            UserMessage(content=[
                TextContentItem(text=f"Look at this image and tell me which of the following descriptions best matches the facial features shown. Please respond with ONLY the matching description, no other text.\n\nPossible descriptions:\n{descriptions_text}"),
                ImageContentItem(image_url=ImageUrl(url=f"data:image/jpeg;base64,{base64_image}"))
            ])
        ],
        max_tokens=800,
        model=model_name
    )
    
    # Extract the result
    return response.choices[0].message.content.strip()

def main():
    parser = argparse.ArgumentParser(description="Analyze facial features in an image based on descriptions")
    parser.add_argument("--descriptions", help="Path to text file containing facial feature descriptions")
    parser.add_argument("--image", help="Path to the image to analyze")
    
    args = parser.parse_args()
    
    try:
        descriptions = read_descriptions(args.descriptions)
        result = analyze_facial_features(args.image, descriptions)
        print(f"Matching facial feature description: {result}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
