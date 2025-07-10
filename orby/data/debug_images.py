import os
import pandas as pd
import argparse
from PIL import Image, ImageDraw
import io

# # Set pandas display options to show full content
# # pd.set_option('display.max_columns', None)
# # pd.set_option('display.max_colwidth', None)
# # pd.set_option('display.width', None)
# # pd.set_option('display.max_rows', None)

# def extract_image_from_message(message):
#     """Extract image data from a message if it contains an image."""
#     if isinstance(message, dict) and 'content' in message:
#         content = message['content']
#         if isinstance(content, list):
#             for item in content:
#                 if isinstance(item, dict) and item.get('type') == 'image':
#                     return item.get('image')
#     return None

# def save_image_with_bbox(image_data, bbox, output_dir, filename, instruction=None, response=None):
#     """Save image with bounding box overlay and predicted click point."""
#     # Handle dict format for image data
#     if isinstance(image_data, dict):
#         actual_bytes = image_data.get('bytes') or image_data.get('data') or image_data.get('image')
#         if actual_bytes:
#             image_data = actual_bytes
    
#     # Create PIL Image from bytes
#     image = Image.open(io.BytesIO(image_data))
    
#     # Create a copy for drawing
#     img_with_bbox = image.copy()
#     draw = ImageDraw.Draw(img_with_bbox)
    
#     # Parse bounding box coordinates - handle numpy array
#     if bbox is not None and hasattr(bbox, '__len__') and len(bbox) >= 4:
#         # Convert to list if it's a numpy array
#         bbox_list = list(bbox) if hasattr(bbox, 'tolist') else bbox
#         x1, y1, x2, y2 = bbox_list[:4]
        
#         # Draw ground truth bounding box in red
#         draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
        
#         # Calculate and draw ground truth center
#         center_x = (x1 + x2) / 2
#         center_y = (y1 + y2) / 2
#         # Draw center point as a small circle
#         radius = 2
#         draw.ellipse([center_x-radius, center_y-radius, center_x+radius, center_y+radius], 
#                     fill="red", outline="darkred", width=1)
        
#         # Add "GT" label near the center
#         draw.text((center_x + 15, center_y - 10), "GT", fill="red", font=None)
    
#     # Extract and draw predicted click coordinates from response
#     if response:
#         import re
#         # Convert numpy array to regular list
#         if hasattr(response, 'tolist'):
#             response = response.tolist()
        
#         response_content = ""
#         print(f"Response value: {response}")
        
#         if isinstance(response, list) and len(response) > 0:
#             response_content = str(response[0]['content'])  # Force string conversion
#         elif isinstance(response, str):
#             response_content = response
            
#         # Extract coordinates from response using regex
#         match = re.search(r'<answer>click\(([0-9.]+),\s*([0-9.]+)\)</answer>', response_content)
#         if match:
#             print(match.groups())
#             pred_x, pred_y = float(match.group(1)), float(match.group(2))
            
#             # Draw predicted click point in blue
#             radius = 2
#             draw.ellipse([pred_x-radius, pred_y-radius, pred_x+radius, pred_y+radius], 
#                         fill="blue", outline="darkblue", width=2)
            
#             # Add "PRED" label near the predicted point
#             draw.text((pred_x + 15, pred_y - 10), "PRED", fill="blue", font=None)
            
#             # Draw a line connecting GT center to predicted point
#             if bbox is not None and hasattr(bbox, '__len__') and len(bbox) >= 4:
#                 draw.line([center_x, center_y, pred_x, pred_y], fill="yellow", width=3)
            
#             print(f"Response center: ({pred_x:.0f}, {pred_y:.0f})")
    
#     # Add instruction text if provided
#     if instruction:
#         # Try to use a larger font
#         try:
#             from PIL import ImageFont
#             font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
#         except:
#             font = None
        
#         # Position text at top of image
#         text_x = 10
#         text_y = 10
        
#         # Draw text background for better visibility
#         if font:
#             text_bbox = draw.textbbox((text_x, text_y), instruction, font=font)
#         else:
#             text_bbox = draw.textbbox((text_x, text_y), instruction)
#         draw.rectangle(text_bbox, fill="yellow", outline="black")
#         draw.text((text_x, text_y), instruction, fill="black", font=font)
    
#     # Save the image with bounding box
#     os.makedirs(output_dir, exist_ok=True)
#     img_with_bbox.save(os.path.join(output_dir, filename))
    
#     return img_with_bbox

# def save_original_image(image_data, output_dir, filename):
#     """Save original image without modifications."""
#     # Handle dict format (likely has 'bytes' or 'data' key)
#     if isinstance(image_data, dict):
#         # Try common keys for image bytes
#         actual_bytes = image_data.get('bytes') or image_data.get('data') or image_data.get('image')
#         if actual_bytes:
#             image_data = actual_bytes
    
#     image = Image.open(io.BytesIO(image_data))
#     os.makedirs(output_dir, exist_ok=True)
#     image.save(os.path.join(output_dir, filename))
#     return image

# def print_image_info(image, filename, response):
#     """Print detailed information about the image."""
#     print(f"\n--- IMAGE INFO: {filename} ---")
#     print(f"Size: {image.size} (width x height)")
#     print(f"Mode: {image.mode}")
#     print(f"Response: {response}")

# def visualize_parquet(parquet_file, save_images=False, original_dir="original_images", bbox_dir="bbox_images", num_examples=20):
#     # Load the parquet file
#     df = pd.read_parquet(parquet_file)
#     print(f"\nParquet file contents ({len(df)} rows):")
#     print("\nColumns:", df.columns.tolist())
    
#     # Print first few rows
#     print("\nFirst few rows:")
#     print(df.head())

    
    
#     idx = 0

#     for idx in range(min(len(df), num_examples)):        
#         # Get the row and extract image data
#         row = df.iloc[idx]
#         image_data = None
        
#         # Try to get image from 'images' column
#         if 'images' in row and row['images'] is not None:
#             images = row['images']
#             if isinstance(images, list) and len(images) > 0:
#                 image_data = images[0]  # Take first image
#             elif hasattr(images, '__len__') and len(images) > 0:
#                 image_data = images[0]  # Handle numpy array
#             elif isinstance(images, bytes):
#                 image_data = images
#         else:
#             print(f"Images column is None or missing")
        
#         if image_data is None:
#             print(f"No image data found for example {idx+1}")

        
#         # Save images if requested and image data is available
#         if image_data is not None:
#             filename = f"example_{idx+1}.png"
            
#             # Save original image
#             original_image = save_original_image(image_data, original_dir, filename)
#             # print_image_info(original_image, f"Original - {filename}", row.get('response', ''))
            
#             # Save image with bounding box
#             extra_info = row.get('extra_info', {})
#             bbox = extra_info.get('bounding_box')
#             print(f"Bounding box: {bbox}")
#             print(f"Center of bbox: {(bbox[0] + bbox[2]) / 2}, {(bbox[1] + bbox[3]) / 2}")
#             if bbox is not None and len(bbox) > 0:
#                 # Get instruction from extra_info
#                 instruction = extra_info.get('question', '') or extra_info.get('answer', '')
#                 print("="*100)
#                 bbox_image = save_image_with_bbox(image_data, bbox, bbox_dir, filename, instruction, row.get('response'))
#                 print_image_info(bbox_image, f"With BBox - {filename}", row.get('response', ''))
#             else:
#                 print("No bounding box found for this example")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--parquet_file", 
#     default="~/data/uground/subtask/0.50k/train/train_part_0000.parquet",
#     help="Path to parquet file to visualize")
#     parser.add_argument("--save_images", action="store_true", help="Save images to directories")
#     parser.add_argument("--original_dir", default="/workspace/verl/orby/data/uground/original_images/", help="Directory to save original images")
#     parser.add_argument("--bbox_dir", default="/workspace/verl/orby/data/uground/bbox_images/", help="Directory to save images with bounding boxes")
#     parser.add_argument("--num_examples", type=int, default=20, help="Number of examples to visualize")
#     args = parser.parse_args()
    
#     visualize_parquet(args.parquet_file, args.save_images, args.original_dir, args.bbox_dir, args.num_examples)


def visualize_parquet(parquet_file):
    # Load the parquet file
    df = pd.read_parquet(parquet_file)
    print(f"\nParquet file contents ({len(df)} rows):")
    print("\nColumns:", df.columns.tolist())

    

    for i in range(15):
        print(f"\nExample {i+1}:")
        print(f"Bounding box: {df.iloc[i]['extra_info']['bounding_box']}")
        print(f"Image size: {df.iloc[i]['width']}x{df.iloc[i]['height']}")
        # Convert image bytes to PIL Image
        image_data = df.iloc[i]['images'][0]
        if isinstance(image_data, dict):
            image_bytes = image_data.get('bytes') or image_data.get('data') or image_data.get('image')
        else:
            image_bytes = image_data
        image = Image.open(io.BytesIO(image_bytes))
        width, height = image.size
        print(f"PIL Image size: {width}x{height}")
        x1, y1, x2, y2 = df.iloc[i]['extra_info']['bounding_box']
        print(f"Center of bbox: {(x1 + x2) / 2:.0f}, {(y1 + y2) / 2:.0f}")
        print(f"Response: {df.iloc[i]['response']}")
        print("-" * 80)


if __name__ == "__main__":
    parquet_file = "~/data/uground/subtask/0.50k/train/train_part_0000.parquet"
    visualize_parquet(parquet_file)