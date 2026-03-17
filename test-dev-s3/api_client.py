#!/usr/bin/env python3
"""
Simple API client to call segmentation API sequentially for multiple images
and visualize the results.
Modified to load images from CloudFront URLs instead of MinIO.
"""

import requests
import json
import sys
import os
from pathlib import Path
from visualize_masks import visualize_result

# API Configuration
API_IP = os.getenv("API_IP", "192.168.80.16")
API_URL = f"http://{API_IP}:8080/api/v2/segmentation/workflow"
AUTH_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJvdHAiOjEyMzQ1NiwiZW1haWwiOiJ0ZXN0QGFpY3ljbGUuYWkiLCJjcmVhdGVkQXQiOiIyMDI0LTA5LTA2VDAyOjQ0OjE3LjQyMloiLCJpYXQiOjE3MjU1OTA2NTcsImV4cCI6MTcyNTYzMzg1N30.5pCsrIG3F1jl2Jjy9cwp8QIRpbK9a6_oD9idOV42rg"

# Request payload template (all params except file_paths)
REQUEST_PAYLOAD = {
    "task_type": "fbf16bcd-7b1b-4d08-9306-d0645118c5cc",
    "trace_id": "85849549498549395324",
    "is_resize_image": True,
    "is_format_output": True,
    "is_upload_mask": True,
    "is_split_mask": True,
    "is_detect_image_overview": False,
    "threshold": 0.5,
    "bucket_name": "ai-insurance",
    "target_folder": "INSURANCE_RESULT",
    "is_classify_car_color": True,
    "is_response_car_model_embedding": False
}


def call_segmentation_api(file_path):
    """
    Call the segmentation API for a single image.
    
    Args:
        file_path: Path to the image file (relative to bucket)
        
    Returns:
        API response as dict, or None if failed
    """
    # Prepare request
    headers = {
        "Authorization": AUTH_TOKEN,
        "Content-Type": "application/json"
    }
    
    payload = REQUEST_PAYLOAD.copy()
    payload["file_paths"] = [file_path]
    
    print(f"\n{'='*60}")
    print(f"Calling API for: {file_path}")
    print(f"{'='*60}")
    
    try:
        # Make API call
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=120  # 2 minutes timeout
        )
        
        # Check response
        response.raise_for_status()
        
        result = response.json()
        print(f"✓ API call successful")
        print(f"Status Code: {response.status_code}")
        
        return result
        
    except requests.exceptions.Timeout:
        print(f"✗ API call timed out for {file_path}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"✗ API call failed for {file_path}: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
        return None
    except json.JSONDecodeError as e:
        print(f"✗ Failed to parse JSON response: {e}")
        return None


def process_images(file_paths):
    """
    Process multiple images sequentially.
    
    Args:
        file_paths: List of image file paths
    """
    print(f"\n{'#'*60}")
    print(f"Starting batch processing for {len(file_paths)} image(s)")
    print(f"{'#'*60}")
    
    results = []
    
    for idx, file_path in enumerate(file_paths, 1):
        print(f"\n[{idx}/{len(file_paths)}] Processing: {file_path}")
        
        # Call API
        api_response = call_segmentation_api(file_path)
        
        if api_response is None:
            print(f"⚠ Skipping visualization for {file_path} due to API error")
            continue
        
        # Extract result from response
        result_list = api_response.get("result", [])
        
        if not result_list:
            print(f"⚠ No results returned for {file_path}")
            continue
        
        # Get first result (should be only one for single image)
        result = result_list[0]
        results.append(result)
        
        # Visualize immediately
        print(f"\nVisualizing result for {file_path}...")
        try:
            visualize_result(result)
        except Exception as e:
            print(f"✗ Visualization failed: {e}")
    
    print(f"\n{'='*60}")
    print(f"✓ Batch processing completed!")
    print(f"Processed: {len(results)}/{len(file_paths)} images")
    print(f"{'='*60}")
    
    return results


def main():
    """Main function"""
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python api_client.py <image1> <image2> ...")
        print("\nExample:")
        print("  python api_client.py BiCBBYyQ9hIDh8DNPg7LF.jpg")
        print("  python api_client.py image1.jpg image2.jpg image3.jpg")
        sys.exit(1)
    
    # Get file paths from command line
    file_paths = sys.argv[1:]
    
    print("="*60)
    print("Segmentation API Client (S3 Version)")
    print("="*60)
    print(f"API Endpoint: {API_URL}")
    print(f"Images to process: {len(file_paths)}")
    for fp in file_paths:
        print(f"  - {fp}")
    
    # Process images
    results = process_images(file_paths)
    
    # Save combined results to JSON file
    output_file = "/home/a4000/Data/ducanhng/CV/Chore/test-dev-s3/api_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"result": results}, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
