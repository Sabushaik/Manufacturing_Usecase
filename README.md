# Manufacturing PPE Detection API

FastAPI-based application for detecting Personal Protective Equipment (PPE) and vehicles in manufacturing environments using:
- **Triton Inference Server** with YOLO for object detection
- **Amazon Bedrock Nova Lite** for vision classification with prompt caching

## Key Features

### 1. Prompt Caching Implementation
The application now implements efficient prompt caching with Bedrock Nova Lite:

#### Startup Phase
- Load 12 sample images (1 from each category: hardhat, goggles, vest, gloves, shoes, forklift, truck, excavator, bull_dozer, cement_mixer, roller, tractor)
- Encode images as base64
- Build system message with sample images
- Mark with `cachePoint` for Bedrock caching
- Store in `SYSTEM_MESSAGES` global variable

#### Request Phase
Each API request sends to Bedrock Nova:
```
┌─────────────────────────────────────┐
│ System Prompt + 12 Sample Images   │ ◄── CACHED by Bedrock
│ (with cachePoint marker)           │
├─────────────────────────────────────┤
│ Crop Image (new each time)         │ ◄── NEW
└─────────────────────────────────────┘
```

### 2. Sample Images
Place 12 sample images in the `samples/` directory (see `samples/README.md` for details):
- 5 PPE categories: hardhat, goggles, vest, gloves, shoes
- 7 vehicle categories: forklift, truck, excavator, bull_dozer, cement_mixer, roller, tractor

### 3. Detection Pipeline
1. **Video Upload**: Upload video to S3 bucket
2. **Detection**: YOLO object detection via Triton
3. **Tracking**: BOTSORT for person and vehicle tracking
4. **Classification**: Bedrock Nova Lite analyzes crops for PPE/vehicle types
5. **Annotation**: Annotated video with PPE status and vehicle labels
6. **Results**: JSON summary with counts and detailed detection data

## API Endpoints

### GET `/`
Returns API information and configuration status

### GET `/health`
Health check for Triton, S3, and Bedrock connections

### POST `/upload-video`
Upload video file to S3 bucket
- **Input**: Video file (multipart/form-data)
- **Output**: S3 URI and presigned URL

### POST `/process-video`
Process video with detection pipeline
- **Input**: JSON with `video_uri` (S3 URI or presigned URL)
- **Output**: Annotated video, JSON results with PPE/vehicle counts

### POST `/test-gpt`
Test Bedrock Nova Lite API connection

## Configuration

### Environment Variables
```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
S3_BUCKET_NAME=your_bucket_name

# Optional: OpenAI Key (legacy, not used with Bedrock)
OPENAI_KEY=your_openai_key
```

### Triton Configuration
- Default URL: `localhost:8000`
- Model: `yolo_person_detection`

## Installation

```bash
# Install dependencies
pip install fastapi uvicorn boto3 pillow opencv-python numpy httpx python-dotenv

# Install Triton client
pip install tritonclient[http]

# Install Ultralytics (for BOTSORT)
pip install ultralytics
```

## Running the Application

```bash
# Run with uvicorn
python main.py

# Or manually
uvicorn main:app --host 0.0.0.0 --port 8092
```

## Response Format

### Video Processing Response
```json
{
  "message": "Successfully processed video",
  "output_s3_uri": "s3://bucket/video_annotated.mp4",
  "output_presigned_url": "https://...",
  "unique_counts": {
    "persons": 5,
    "vehicles": 3
  },
  "per_person_ppe_summary": {
    "1": {
      "hardhat": true,
      "goggles": true,
      "safety_vest": true,
      "gloves": true,
      "shoes": true,
      "PPE": true
    }
  },
  "ppe_summary": {
    "hard_hat": 5,
    "goggles": 4,
    "safety_vest": 5,
    "gloves": 3,
    "safety_shoes": 4,
    "persons_detected": 5,
    "list_of_vehicles": ["forklift", "truck"]
  },
  "vehicle_counts": {
    "forklift": 2,
    "truck": 1
  },
  "timing": { ... }
}
```

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Client    │────▶│   FastAPI    │────▶│   Triton    │
│             │     │   (main.py)  │     │   (YOLO)    │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                           │
                    ┌──────▼──────┐
                    │   Bedrock   │
                    │  Nova Lite  │
                    │  (Cached)   │
                    └─────────────┘
                           │
                           │
                    ┌──────▼──────┐
                    │     S3      │
                    │  (Storage)  │
                    └─────────────┘
```

## Features Preserved

All existing functionalities remain intact:
- ✅ Video upload to S3
- ✅ Presigned URL support
- ✅ S3 URI support
- ✅ Web format video conversion (FFmpeg)
- ✅ JSON result export
- ✅ Annotated video output
- ✅ PPE detection and counting
- ✅ Vehicle detection and classification
- ✅ Person tracking with BOTSORT
- ✅ Multi-threaded processing

## Prompt Caching Benefits

1. **Cost Reduction**: System prompt + 12 images cached, only crop images charged per request
2. **Performance**: Faster inference due to cached context
3. **Consistency**: Same sample images used across all requests
4. **Scalability**: Efficient handling of multiple requests

## Notes

- Sample images are loaded once at startup
- Missing sample images trigger warnings but don't prevent operation
- Bedrock caching is automatic when `cachePoint` marker is present
- Cache is managed by Bedrock infrastructure (no manual cache management needed)
