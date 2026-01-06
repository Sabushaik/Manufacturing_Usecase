"""
Example usage of the video processing API with different modelId options
"""

import requests
import json

API_BASE_URL = "http://localhost:8092"

def test_process_video_default_model():
    """Test video processing with default model (nova-lite)"""
    payload = {
        "video_uri": "s3://my-bucket/videos/manufacturing.mp4"
    }
    
    response = requests.post(
        f"{API_BASE_URL}/process-video",
        json=payload
    )
    
    print("Test 1: Default Model (nova-lite)")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Model Used: {result['config']['vision_model']}")
        print(f"Persons Detected: {result['unique_counts']['persons']}")
        print(f"Vehicles Detected: {result['unique_counts']['vehicles']}")
    else:
        print(f"Error: {response.text}")
    print("-" * 80)


def test_process_video_nova_pro():
    """Test video processing with nova-pro model"""
    payload = {
        "video_uri": "s3://my-bucket/videos/manufacturing.mp4",
        "modelId": "amazon.nova-pro-v1:0"
    }
    
    response = requests.post(
        f"{API_BASE_URL}/process-video",
        json=payload
    )
    
    print("Test 2: Nova Pro Model")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Model Used: {result['config']['vision_model']}")
        print(f"Message: {result['message']}")
    else:
        print(f"Error: {response.text}")
    print("-" * 80)


def test_process_video_nova_micro():
    """Test video processing with nova-micro model"""
    payload = {
        "video_uri": "s3://my-bucket/videos/manufacturing.mp4",
        "modelId": "amazon.nova-micro-v1:0"
    }
    
    response = requests.post(
        f"{API_BASE_URL}/process-video",
        json=payload
    )
    
    print("Test 3: Nova Micro Model")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Model Used: {result['config']['vision_model']}")
    else:
        print(f"Error: {response.text}")
    print("-" * 80)


def test_process_video_with_presigned_url():
    """Test video processing with presigned URL and custom model"""
    payload = {
        "video_uri": "https://my-bucket.s3.amazonaws.com/videos/video.mp4?AWSAccessKeyId=...",
        "modelId": "amazon.nova-pro-v1:0"
    }
    
    response = requests.post(
        f"{API_BASE_URL}/process-video",
        json=payload
    )
    
    print("Test 4: Presigned URL with Nova Pro")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Input Type: {result['input_type']}")
        print(f"Model Used: {result['config']['vision_model']}")
    else:
        print(f"Error: {response.text}")
    print("-" * 80)


def test_invalid_model():
    """Test with invalid model ID"""
    payload = {
        "video_uri": "s3://my-bucket/videos/manufacturing.mp4",
        "modelId": "invalid-model-id"
    }
    
    response = requests.post(
        f"{API_BASE_URL}/process-video",
        json=payload
    )
    
    print("Test 5: Invalid Model ID (should fail)")
    print(f"Status: {response.status_code}")
    print(f"Error: {response.text}")
    print("-" * 80)


def test_health_check():
    """Test health check endpoint"""
    response = requests.get(f"{API_BASE_URL}/health")
    
    print("Health Check")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Triton Connected: {result['triton_connected']}")
        print(f"Bedrock Connected: {result['bedrock_connected']}")
        print(f"Vision Model: {result['vision_model']}")
    print("-" * 80)


if __name__ == "__main__":
    print("=" * 80)
    print("Video Processing API - Model Selection Tests")
    print("=" * 80)
    print()
    
    # Run health check first
    test_health_check()
    
    # Note: Uncomment the tests you want to run
    # These require a running API server and valid S3 URIs
    
    # test_process_video_default_model()
    # test_process_video_nova_pro()
    # test_process_video_nova_micro()
    # test_process_video_with_presigned_url()
    # test_invalid_model()
    
    print()
    print("Tests completed!")
