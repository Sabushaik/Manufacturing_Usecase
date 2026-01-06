# API Changes Summary

## What Changed

### 1. New Optional Parameter: `modelId`
The `/process-video` endpoint now accepts an optional `modelId` parameter to specify which Amazon Bedrock Nova model to use for vision analysis.

### 2. Default Behavior
If no `modelId` is provided, the system uses `amazon.nova-lite-v1:0` by default.

### 3. Supported Models
- `amazon.nova-lite-v1:0` (default) - Fast and cost-effective
- `amazon.nova-pro-v1:0` - Higher accuracy
- `amazon.nova-micro-v1:0` - Ultra-fast processing

## API Request Changes

### Before (Old API):
```json
{
  "video_uri": "s3://bucket/videos/video.mp4"
}
```

### After (New API):
```json
{
  "video_uri": "s3://bucket/videos/video.mp4",
  "modelId": "amazon.nova-pro-v1:0"
}
```

### Backward Compatibility:
The old format still works! If you don't provide `modelId`, it defaults to `amazon.nova-lite-v1:0`.

## Implementation Details

### Files Modified:
1. **main.py** - Core implementation
   - Updated `VideoInput` model with optional `modelId` field
   - Modified `call_gpt_single()` to accept `modelId` parameter
   - Updated `process_crops_async()` to pass `modelId` through
   - Modified `process_video_with_gpt_pipeline()` to accept and use `modelId`
   - Updated `/process-video` endpoint to extract and use `modelId` from request
   - Enhanced logging to show which model is being used
   - Updated response to include the `modelId` used for processing

### Files Added:
1. **README.md** - Comprehensive API documentation
2. **requirements.txt** - Python dependencies
3. **test_api.py** - Example usage and tests
4. **CHANGES.md** - This file

## Code Flow

```
User Request
    ↓
VideoInput Model (validates modelId)
    ↓
/process-video endpoint (extracts modelId)
    ↓
process_video_with_gpt_pipeline (passes modelId)
    ↓
process_crops_async (passes modelId to workers)
    ↓
call_gpt_single (uses modelId in Bedrock API call)
    ↓
Amazon Bedrock Nova API (processes with specified model)
    ↓
Response (includes modelId used)
```

## Key Features

### 1. Validation
The `modelId` parameter is validated to ensure only supported models are used:
```python
allowed_models = [
    "amazon.nova-lite-v1:0",
    "amazon.nova-pro-v1:0",
    "amazon.nova-micro-v1:0"
]
```

### 2. Logging
Enhanced logging throughout the pipeline shows which model is being used:
```
🤖 Model: amazon.nova-pro-v1:0
🤖 Phase 2: Bedrock amazon.nova-pro-v1:0 Vision Analysis...
✅ Phase 2 complete: Bedrock amazon.nova-pro-v1:0 analysis done in 45.23s
```

### 3. Response
The API response now includes the model used for transparency:
```json
{
  "message": "Successfully processed video with YOLO + Bedrock amazon.nova-pro-v1:0 Vision pipeline",
  "config": {
    "vision_model": "amazon.nova-pro-v1:0",
    "vision_provider": "Amazon Bedrock"
  }
}
```

## Testing

Use the provided `test_api.py` to test different models:

```bash
python test_api.py
```

## Migration Guide

### For Existing Users:
No changes required! Your existing code will continue to work with the default `amazon.nova-lite-v1:0` model.

### For New Features:
Simply add the `modelId` field to your request to use a different model:

```python
import requests

response = requests.post(
    "http://localhost:8092/process-video",
    json={
        "video_uri": "s3://bucket/video.mp4",
        "modelId": "amazon.nova-pro-v1:0"  # New parameter
    }
)
```

## Error Handling

Invalid model IDs will return a validation error:
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
