# Quick Start Guide

## TL;DR - What Was Implemented

✅ **Prompt caching with 12 sample images** for Bedrock Nova Lite
✅ **All existing functionalities preserved** (upload, S3, presigned URLs, video conversion, JSON responses)
✅ **Comprehensive documentation** and verification tests

## Files You Need to Know

### Core Application
- **`main.py`**: Main FastAPI application with prompt caching

### Documentation
- **`README.md`**: Overview and architecture
- **`SETUP.md`**: Step-by-step setup instructions
- **`samples/README.md`**: Sample image requirements
- **`IMPLEMENTATION_SUMMARY.md`**: Technical details

### Configuration
- **`requirements.txt`**: Python dependencies
- **`.env`**: Environment variables (you need to create this)

### Testing
- **`test_caching_structure.py`**: Verification tests (already passing ✅)

## What You Need to Do

### 1. Add Sample Images (12 Required)

Place these files in the repository:

**PPE (5 images):**
```
samples/hardhat/hardhat1.jpg
samples/goggles/goggles1.jpg
samples/vest/safety_vest1.jpg
samples/gloves/gloves1.jpg
samples/shoes/shoes1.jpg
```

**Vehicles (7 images):**
```
samples/forklift/forklift1.jpg
samples/truck/truck1.jpg
samples/excavator/excavator1.jpg
samples/bull_dozer/bull_dozer1.jpg
samples/cement_mixer/cement_mixer1.jpg
samples/roller/roller1.jpg
samples/tracktor/tracktor1.jpg
```

**Important**: Use clear, representative examples. JPEG format, < 1MB each.

### 2. Create `.env` File

```bash
AWS_ACCESS_KEY_ID=your_access_key_id_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_REGION=us-east-1
S3_BUCKET_NAME=your_bucket_name_here
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Triton Server

Ensure YOLO model is running on `localhost:8000`

### 5. Run Application

```bash
python main.py
```

Look for this message:
```
✅ Loaded 12 sample images for Bedrock Nova Lite prompt caching
   Sample categories: hardhat, goggles, vest, gloves, shoes, forklift, truck, excavator, bull_dozer, cement_mixer, roller, tractor
   Cache point marker added to system messages for efficient reuse
```

## How It Works

### Before (No Caching)
❌ Every request sent system prompt + examples + crop image
❌ High token costs
❌ Slower inference

### After (With Caching) ✅
```
STARTUP: Load 12 images → Build cached system messages

REQUEST 1: Send system (cached) + crop1 → Bedrock caches system
REQUEST 2: Send system (cached) + crop2 → Bedrock reuses cache ⚡
REQUEST 3: Send system (cached) + crop3 → Bedrock reuses cache ⚡
REQUEST N: Send system (cached) + cropN → Bedrock reuses cache ⚡
```

**Benefits:**
- 💰 Lower costs (system content cached)
- ⚡ Faster responses
- 📊 Consistent examples across requests

## API Usage

### Upload Video
```bash
curl -X POST http://localhost:8092/upload-video \
  -F "video=@video.mp4"
```

### Process Video
```bash
curl -X POST http://localhost:8092/process-video \
  -H "Content-Type: application/json" \
  -d '{"video_uri": "s3://bucket/video.mp4"}'
```

### Test Bedrock
```bash
curl -X POST http://localhost:8092/test-gpt
```

## Verification

Run the structure test:
```bash
python test_caching_structure.py
```

Expected output:
```
✅ ALL TESTS PASSED!
```

## Troubleshooting

### Sample Images Not Found
- Check file paths match exactly (including folder names)
- Verify images are JPEG format
- Ensure files are in correct directories

### Bedrock Errors
- Verify AWS credentials in `.env`
- Check IAM permissions for Bedrock
- Ensure region supports Nova Lite (us-east-1 recommended)

### Triton Connection Failed
- Check Triton is running: `curl http://localhost:8000/v2/health/ready`
- Verify model is loaded
- Check Triton logs

## Key Features Preserved

✅ Video upload to S3
✅ S3 URI and presigned URL support
✅ FFmpeg video conversion
✅ JSON results export
✅ Annotated video output
✅ PPE detection (5 types)
✅ Vehicle classification (7 types)
✅ Person/vehicle tracking
✅ All response formats
✅ Error handling
✅ Health checks

## Support

For detailed information, see:
- **Architecture**: `README.md`
- **Setup**: `SETUP.md`
- **Implementation**: `IMPLEMENTATION_SUMMARY.md`
- **Sample Images**: `samples/README.md`

## Summary

**What changed**: Added prompt caching with 12 sample images for efficiency
**What didn't change**: Everything else (all existing features work exactly the same)
**What you need**: 12 sample images + AWS credentials + Triton server
**What you get**: Faster, cheaper Bedrock inference with same great results
