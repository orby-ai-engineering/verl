import pyarrow.parquet as pq
import io
from PIL import Image, ImageDraw
import os
import re

def extract_images(parquet_file, num_images=10000, output_dir="/workspace/verl/orby/data/screenspot_v1/"):
    """
    Extract images from ScreenSpot parquet file and draw bounding boxes with coordinates
    """
    parquet_file_expanded = parquet_file.replace('~', '/root')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the parquet file and iterate through batches
    pf = pq.ParquetFile(parquet_file_expanded)
    batch_reader = pf.iter_batches(batch_size=1, columns=['images', 'responses', 'extra_info'])
    
    extracted_count = 0
    
    for i, batch in enumerate(batch_reader):
        if extracted_count >= num_images:
            break
            
        images_col = batch.column('images')
        responses_col = batch.column('responses')
        extra_info_col = batch.column('extra_info')
        
        if len(images_col) > 0 and images_col[0].is_valid:
            image_data = images_col[0].as_py()
            responses = responses_col[0].as_py() if len(responses_col) > 0 and responses_col[0].is_valid else None
            extra_info = extra_info_col[0].as_py() if len(extra_info_col) > 0 and extra_info_col[0].is_valid else None
            
            if image_data and isinstance(image_data, list) and len(image_data) > 0:
                first_item = image_data[0]
                
                if isinstance(first_item, dict) and 'bytes' in first_item:
                    image_bytes = first_item['bytes']
                    
                    try:
                        image = Image.open(io.BytesIO(image_bytes))
                        draw = ImageDraw.Draw(image)
                        
                        # Draw ground truth bounding box in red
                        if extra_info and 'bounding_box' in extra_info:
                            bbox = extra_info['bounding_box']
                            x1, y1, x2, y2 = bbox
                            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                            # Draw GT center point
                            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                            draw.ellipse([center_x-3, center_y-3, center_x+3, center_y+3], fill="red")
                            draw.text((center_x+10, center_y-10), "GT", fill="red")
                        
                        # Extract and draw predicted coordinates in blue
                        if responses and len(responses) > 0:
                            response_text = responses[0]
                            match = re.search(r'click\((\d+\.?\d*),\s*(\d+\.?\d*)\)', response_text)
                            if match:
                                pred_x, pred_y = float(match.group(1)), float(match.group(2))
                                draw.ellipse([pred_x-3, pred_y-3, pred_x+3, pred_y+3], fill="blue")
                                draw.text((pred_x+10, pred_y-10), "PRED", fill="blue")
                        
                        # Draw question text near the bounding box
                        if extra_info and 'question' in extra_info and 'bounding_box' in extra_info:
                            question = extra_info['question']
                            bbox = extra_info['bounding_box']
                            x1, y1, x2, y2 = bbox
                            # Position text above the bounding box
                            text_x = x1 + 30 # 30 pixels right of the bounding box
                            text_y = y1 + 30  # 30 pixels below the bounding box
                            # Draw text background for better visibility
                            text_bbox = draw.textbbox((text_x, text_y), question)
                            draw.rectangle([text_bbox[0]-5, text_bbox[1]-5, text_bbox[2]+5, text_bbox[3]+5],
                                         fill="yellow", outline="black")
                            draw.text((text_x, text_y), question, fill="black")
                        output_path = os.path.join(output_dir, f"image_{i:04d}.png")
                        image.save(output_path)
                        extracted_count += 1
                        print(f"Saved image {extracted_count}: {output_path}")
                        
                    except Exception as e:
                        print(f"Failed to process image {i}: {e}")
    
    print(f"\nExtracted {extracted_count} images to {output_dir}")
    return extracted_count

if __name__ == "__main__":
    parquet_file = "~/data/screenspot_subtask/result-test-output-1.parquet"
    extract_images(parquet_file, num_images=100) 