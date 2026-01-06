# Quick Start Guide

## Overview
This guide shows you how to quickly test the new modelId parameter feature.

## Prerequisites
1. Python 3.8+
2. AWS credentials with Bedrock access
3. S3 bucket with videos
4. Running Triton inference server

## Installation

```bash
# Clone the repository
git clone https://github.com/Sabushaik/Manufacturing_Usecase.git
cd Manufacturing_Usecase

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export S3_BUCKET_NAME="your-input-bucket"
export OUTPUT_S3_BUCKET_NAME="your-output-bucket"
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_REGION="us-east-1"
```

## Running the Server

```bash
python main.py
```

The server will start on `http://localhost:8092`

## Testing the API

### 1. Health Check

```bash
curl http://localhost:8092/health
```

Expected response:
```json
{
  "status": "healthy",
  "triton_connected": true,
  "bedrock_connected": true,
  "vision_model": "amazon.nova-lite-v1:0"
}
```

### 2. Process Video with Default Model (Nova Lite)

```bash
curl -X POST http://localhost:8092/process-video \
  -H "Content-Type: application/json" \
  -d '{
    "video_uri": "s3://your-bucket/videos/test.mp4"
  }'
```

### 3. Process Video with Nova Pro

```bash
curl -X POST http://localhost:8092/process-video \
  -H "Content-Type: application/json" \
  -d '{
    "video_uri": "s3://your-bucket/videos/test.mp4",
    "modelId": "amazon.nova-pro-v1:0"
  }'
```

### 4. Process Video with Nova Micro

```bash
curl -X POST http://localhost:8092/process-video \
  -H "Content-Type: application/json" \
  -d '{
    "video_uri": "s3://your-bucket/videos/test.mp4",
    "modelId": "amazon.nova-micro-v1:0"
  }'
```

### 5. Using Presigned URL

```bash
curl -X POST http://localhost:8092/process-video \
  -H "Content-Type: application/json" \
  -d '{
    "video_uri": "https://your-bucket.s3.amazonaws.com/videos/test.mp4?X-Amz-...",
    "modelId": "amazon.nova-lite-v1:0"
  }'
```

## Python Example

```python
import requests

# Using default model
response = requests.post(
    "http://localhost:8092/process-video",
    json={
        "video_uri": "s3://your-bucket/videos/test.mp4"
    }
)

print(f"Status: {response.status_code}")
print(f"Model Used: {response.json()['config']['vision_model']}")
print(f"Persons: {response.json()['unique_counts']['persons']}")
print(f"Vehicles: {response.json()['unique_counts']['vehicles']}")

# Using nova-pro model
response = requests.post(
    "http://localhost:8092/process-video",
    json={
        "video_uri": "s3://your-bucket/videos/test.mp4",
        "modelId": "amazon.nova-pro-v1:0"
    }
)

print(f"Model Used: {response.json()['config']['vision_model']}")
```

## Expected Response

```json
{
  "message": "Successfully processed video with YOLO + Bedrock amazon.nova-lite-v1:0 Vision pipeline",
  "input_uri": "s3://bucket/videos/test.mp4",
  "input_type": "s3_uri",
  "output_s3_uri": "s3://output-bucket/videos/test_annotated.mp4",
  "output_presigned_url": "https://...",
  "results_json_uri": "s3://output-bucket/videos/test_results.json",
  "config": {
    "model_name": "yolo_person_detection",
    "triton_server": "localhost:8000",
    "vision_model": "amazon.nova-lite-v1:0",
    "vision_provider": "Amazon Bedrock"
  },
  "unique_counts": {
    "persons": 5,
    "vehicles": 2
  },
  "ppe_summary": {
    "hard_hat": 4,
    "goggles": 3,
    "safety_vest": 5,
    "gloves": 2,
    "safety_shoes": 5,
    "persons_detected": 5,
    "list_of_vehicles": ["forklift", "truck"]
  },
  "timing": {
    "s3_download_duration": 2.5,
    "processing_duration": 45.3,
    "s3_upload_video_duration": 3.2,
    "total_duration": 51.0
  }
}
```

## Verifying Model Selection

Check the response to confirm the correct model was used:

```python
response = requests.post(...)
model_used = response.json()['config']['vision_model']
print(f"Model used: {model_used}")

# Should print: "Model used: amazon.nova-pro-v1:0" if you specified nova-pro
```

## Error Handling

### Invalid Model ID
```bash
curl -X POST http://localhost:8092/process-video \
  -H "Content-Type: application/json" \
  -d '{
    "video_uri": "s3://bucket/video.mp4",
    "modelId": "invalid-model"
  }'
```

Response:
```json
{
  "detail": [
    {
      "loc": ["body", "modelId"],
      "msg": "modelId must be one of ['amazon.nova-lite-v1:0', 'amazon.nova-pro-v1:0', 'amazon.nova-micro-v1:0']",
      "type": "value_error"
    }
  ]
}
```

## Using the Test Script

```bash
# Edit test_api.py to uncomment the tests you want to run
# Then run:
python test_api.py
```

## Troubleshooting

### Server won't start
- Check that all environment variables are set
- Verify Triton server is running on localhost:8000
- Check AWS credentials have Bedrock access

### Processing fails
- Verify S3 bucket permissions
- Check AWS region matches your Bedrock setup
- Ensure video format is supported (mp4, avi, mov, etc.)

### Wrong model used
- Check the response to see which model was actually used
- Verify modelId spelling is correct (case-sensitive)
- Ensure modelId is in the allowed list

## Model Comparison

| Model | Speed | Accuracy | Cost | Use Case |
|-------|-------|----------|------|----------|
| nova-lite-v1:0 | Fast | Good | Low | Standard processing |
| nova-pro-v1:0 | Moderate | Excellent | High | Critical safety |
| nova-micro-v1:0 | Very Fast | Basic | Very Low | High throughput |

## Next Steps

1. Test with your own videos
2. Compare results between different models
3. Choose the best model for your use case
4. Integrate into your workflow

## Support

For issues or questions:
- Check CHANGES.md for detailed implementation info
- Review README.md for comprehensive documentation
- See SUMMARY.md for complete overview
