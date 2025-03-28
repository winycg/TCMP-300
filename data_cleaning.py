import os
import shutil
import torch
import clip
import argparse
from PIL import Image

# Installation commands (if needed):
# conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
# pip install ftfy regex tqdm
# pip install git+https://github.com/openai/CLIP.git


def parse_option():
    parser = argparse.ArgumentParser('CLIP Image Search Script')
    parser.add_argument(
        '--image2image', 
        type=str, 
        default="False",
        help='Set mode: True for image-to-image search, False for text-to-image search.If image2image is True, use the specific image to serach for target images, else use text to search for target images'
    )
    parser.add_argument(
        "--base_image_path",
        type=str,
        default = None,
        help="Path to the base image.",
    )
    parser.add_argument(
        "--search_text",
        type=str,
        default="a photo of brown slices or tubers",
        help="Text description for text-to-image search.",
    )
    parser.add_argument(
        "--source_folder",
        type=str,
        required=True,
        help="Path to the source folder containing images.",
    )
    parser.add_argument(
        "--target_folder",
        type=str,
        required=True,
        help="Path to the target folder where matching images will be moved.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.662,
        help="Similarity threshold for matching images. It can be adjusted accoding to actural needs",
    )
    return parser.parse_args()


def find_and_move_matching_images(source_folder, target_folder, image2image, model, preprocess, text, threshold, base_image_path = None):
    """
    Find and move images that match the search query (text or image).
    
    Args:
        source_folder (str): Path to the folder containing images to be cleaned.
        target_folder (str): Path to the folder where matching images will be moved.
        image2image (bool): If True, perform image-to-image search; else, text-to-image search.
        model: CLIP model for encoding images and text.
        preprocess: CLIP preprocessing function for images.
        text: Tokenized text input for text-to-image search.
        base_image_path: Reference image for image-to-image search.
    """
    # Create target folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for filename in os.listdir(source_folder):
        # Process only image files
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(source_folder, filename)
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

            if image2image == "True":
                # Use specific image as the reference for image-to-image search
                if base_image_path is None:
                    raise ValueError("For image-to-image mode, base_image_path parameter must be specified with a valid image path.")
                
                base_image=preprocess(Image.open(base_image_path)).unsqueeze(0).to(device)

                # Encode images
                with torch.no_grad():
                    image_features = model.encode_image(image)
                    base_features = model.encode_image(base_image)

            else:
                # Encode image and text
                with torch.no_grad():
                    image_features = model.encode_image(image)
                    base_features = model.encode_text(text)
                    
            # calculate similarity between image and desciption (image or text) 
            similarity = torch.cosine_similarity(image_features, base_features)
            
            # if the similarity is greater than the threshold, the coresponding image is regarded as the target image
            if similarity.item() > threshold: 
                target_path = os.path.join(target_folder, filename)
                shutil.move(image_path, target_path)
                print(f"Moved {filename} to {target_folder}")


if __name__ == "__main__": 
    # Parse command-line arguments
    args = parse_option()

    # Load CLIP model and set device (GPU if available, else CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    if not args.image2image:
        # Define the image description
        text = clip.tokenize([args.search_text]).to(device)
    else:
        text = None

    # Find and remove the target images
    find_and_move_matching_images(
        source_folder=args.source_folder,
        target_folder=args.target_folder,
        image2image=args.image2image,
        model=model,
        preprocess=preprocess,
        text=text,
        threshold=args.threshold,
        base_image_path=args.base_image_path
    )

    print("finish")
