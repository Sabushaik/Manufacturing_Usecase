# Manufacturing Use Case - Video Processing API

This API processes manufacturing videos using YOLO detection and Amazon Bedrock Nova vision models for PPE (Personal Protective Equipment) detection and vehicle classification.

## Features

- Video processing with YOLO object detection
- PPE detection (hardhat, goggles, safety vest, gloves, shoes)
- Vehicle detection and classification
- Support for multiple Amazon Bedrock Nova models
- S3 integration for video storage
- Real-time video annotation

## API Endpoints

### GET /
Returns API information and configuration status.

### GET /health
Health check endpoint with service status information.

### POST /upload-video
Upload a video file to S3 for processing.

**Request:**
- Form data with video file

**Response:**
```json
{
  "message": "Successfully uploaded",
  "filename": "videos/20240106_123456_video.mp4",
  "s3_uri": "s3://bucket/videos/20240106_123456_video.mp4",
  "presigned_url": "https://...",
  "file_size_mb": 25.4
}
```

### POST /process-video
Process a video for PPE detection and vehicle classification.

**Request:**
```json
{
  "video_uri": "s3://bucket/videos/video.mp4",
  "modelId": "amazon.nova-lite-v1:0"
}
```

**Supported Models:**
- `amazon.nova-lite-v1:0` (default) - Fast and cost-effective
- `amazon.nova-pro-v1:0` - Higher accuracy
- `amazon.nova-micro-v1:0` - Ultra-fast processing

**Parameters:**
- `video_uri` (required): S3 URI (s3://...) or presigned URL (https://...)
- `modelId` (optional): Amazon Bedrock Nova model ID. Defaults to `amazon.nova-lite-v1:0`

**Response:**
```json
{
  "message": "Successfully processed video with YOLO + Bedrock amazon.nova-lite-v1:0 Vision pipeline",
  "input_uri": "s3://bucket/videos/video.mp4",
  "output_s3_uri": "s3://bucket/videos/video_annotated.mp4",
  "output_presigned_url": "https://...",
  "results_json_uri": "s3://bucket/videos/video_results.json",
  "config": {
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
    "persons_detected": 5
  }
}
```

## Usage Examples

### Process video with default model (nova-lite):
```bash
curl -X POST http://localhost:8092/process-video \
  -H "Content-Type: application/json" \
  -d '{
    "video_uri": "s3://my-bucket/videos/manufacturing.mp4"
  }'
```

### Process video with nova-pro model:
```bash
curl -X POST http://localhost:8092/process-video \
  -H "Content-Type: application/json" \
  -d '{
    "video_uri": "s3://my-bucket/videos/manufacturing.mp4",
    "modelId": "amazon.nova-pro-v1:0"
  }'
```

### Process video with presigned URL:
```bash
curl -X POST http://localhost:8092/process-video \
  -H "Content-Type: application/json" \
  -d '{
    "video_uri": "https://my-bucket.s3.amazonaws.com/videos/video.mp4?...",
    "modelId": "amazon.nova-lite-v1:0"
  }'
```

## Environment Variables

Required environment variables:
- `S3_BUCKET_NAME`: Input S3 bucket name
- `OUTPUT_S3_BUCKET_NAME`: Output S3 bucket name
- `AWS_ACCESS_KEY_ID`: AWS access key
- `AWS_SECRET_ACCESS_KEY`: AWS secret key
- `AWS_REGION`: AWS region (default: us-east-1)

## Model Selection Guide

### amazon.nova-lite-v1:0 (Default)
- **Best for**: Standard processing, cost-effective
- **Speed**: Fast
- **Accuracy**: Good
- **Use case**: Regular PPE detection and vehicle classification

### amazon.nova-pro-v1:0
- **Best for**: High-accuracy requirements
- **Speed**: Moderate
- **Accuracy**: Excellent
- **Use case**: Critical safety applications, detailed analysis

### amazon.nova-micro-v1:0
- **Best for**: Real-time processing, large volumes
- **Speed**: Very fast
- **Accuracy**: Basic
- **Use case**: Quick screening, high-throughput scenarios

## Architecture

1. **Video Upload**: Videos are uploaded to S3
2. **Detection**: YOLO model on Triton server detects persons and vehicles
3. **Tracking**: BOTSORT tracker maintains object IDs across frames
4. **Classification**: Amazon Bedrock Nova models analyze cropped images for PPE and vehicle types
5. **Annotation**: Processed video is annotated with detection results
6. **Output**: Annotated video and JSON results are uploaded to S3

## Dependencies

- FastAPI
- OpenCV
- Boto3 (AWS SDK)
- Triton Inference Client
- Ultralytics (YOLO tracking)
- Pydantic

## Running the Server

```bash
python main.py
```

The server will start on port 8092.
