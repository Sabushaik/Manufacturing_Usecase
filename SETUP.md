# Setup Guide

## Prerequisites

1. **Python 3.8+**
2. **Triton Inference Server** running on `localhost:8000` with YOLO model
3. **AWS Account** with:
   - S3 bucket access
   - Bedrock access (for Nova Lite model)
   - Valid credentials
4. **Sample Images** (12 images - see below)

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Configure Environment Variables

Create a `.env` file in the project root:

```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
AWS_REGION=us-east-1
S3_BUCKET_NAME=your_bucket_name
```

## Step 3: Prepare Sample Images

Place 12 sample images in the `samples/` directory structure:

### PPE Categories (5 images):
```
samples/hardhat/hardhat1.jpg
samples/goggles/goggles1.jpg
samples/vest/safety_vest1.jpg
samples/gloves/gloves1.jpg
samples/shoes/shoes1.jpg
```

### Vehicle Categories (7 images):
```
samples/forklift/forklift1.jpg
samples/truck/truck1.jpg
samples/excavator/excavator1.jpg
samples/bull_dozer/bull_dozer1.jpg
samples/cement_mixer/cement_mixer1.jpg
samples/roller/roller1.jpg
samples/tracktor/tracktor1.jpg  # Note: folder is 'tracktor' not 'tractor'
```

**Important Notes:**
- Images should be clear, representative examples of each category
- Recommended resolution: 640x640 or similar
- Format: JPEG
- File size: < 1MB per image recommended

## Step 4: Verify Triton Server

Ensure Triton Inference Server is running:

```bash
# Check if Triton is accessible
curl http://localhost:8000/v2/health/ready
```

Expected response: `{"ready": true}`

## Step 5: Start the Application

```bash
python main.py
```

Or with uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8092
```

## Step 6: Verify Setup

### Check API Status
```bash
curl http://localhost:8092/
```

### Check Health
```bash
curl http://localhost:8092/health
```

### Test Bedrock Connection
```bash
curl -X POST http://localhost:8092/test-gpt
```

## Application Startup

When the application starts successfully, you should see:

```
✅ S3 client initialized
✅ Bedrock client initialized
✅ Connected to Triton server at localhost:8000
✅ Using model: yolo_person_detection
✅ Loaded 12 sample images for Bedrock Nova Lite prompt caching
   Sample categories: hardhat, goggles, vest, gloves, shoes, forklift, truck, excavator, bull_dozer, cement_mixer, roller, tractor
   Cache point marker added to system messages for efficient reuse
```

## Testing the Pipeline

### 1. Upload a Video

```bash
curl -X POST http://localhost:8092/upload-video \
  -F "video=@/path/to/your/video.mp4"
```

Response will include:
- `s3_uri`: S3 location of uploaded video
- `presigned_url`: Direct download link (24-hour expiry)

### 2. Process the Video

Using S3 URI:
```bash
curl -X POST http://localhost:8092/process-video \
  -H "Content-Type: application/json" \
  -d '{"video_uri": "s3://your-bucket/videos/20231230_120000_video.mp4"}'
```

Or using Presigned URL:
```bash
curl -X POST http://localhost:8092/process-video \
  -H "Content-Type: application/json" \
  -d '{"video_uri": "https://your-bucket.s3.amazonaws.com/..."}'
```

## Understanding Prompt Caching

### What Gets Cached?
- System prompt text (PPE/vehicle detection rules)
- 12 sample images (1 per category)
- Total cached content: ~13 text blocks + 12 image blocks

### What's Sent Each Time?
- New crop image from detected person/vehicle
- Analysis request

### Benefits
1. **Cost Reduction**: Cached content isn't charged repeatedly
2. **Performance**: Faster inference due to cached context
3. **Consistency**: Same examples across all requests
4. **Scalability**: Efficient for high-volume processing

### Cache Behavior
- Cache type: `ephemeral` (managed by Bedrock)
- Cache lifetime: Managed automatically by AWS
- Cache key: Based on content, automatically handled

## Troubleshooting

### Sample Images Not Loading
```
⚠️ Sample image not found: samples/hardhat/hardhat1.jpg
```
**Solution**: Ensure all 12 sample images are in the correct locations

### Triton Connection Failed
```
❌ Failed to initialize Triton client
```
**Solution**: 
1. Check Triton is running: `curl http://localhost:8000/v2/health/ready`
2. Verify model is loaded: Check Triton logs
3. Ensure correct URL in configuration

### Bedrock Client Not Initialized
```
❌ Failed to initialize Bedrock client
```
**Solution**:
1. Verify AWS credentials in `.env`
2. Check IAM permissions for Bedrock access
3. Ensure correct region (us-east-1 recommended for Nova)
4. Test credentials: `aws sts get-caller-identity`

### S3 Upload Failed
```
S3 upload failed: Access Denied
```
**Solution**:
1. Verify S3 bucket exists
2. Check IAM permissions for S3 write access
3. Ensure bucket name is correct in `.env`

## Performance Tips

1. **FFmpeg**: Install FFmpeg for web-compatible video conversion
   ```bash
   # Ubuntu/Debian
   sudo apt-get install ffmpeg
   
   # macOS
   brew install ffmpeg
   ```

2. **GPU**: Ensure Triton server has GPU access for faster YOLO inference

3. **Batch Processing**: Process multiple videos sequentially for better throughput

4. **Network**: Use VPC endpoints for S3/Bedrock for lower latency

## Monitoring

Check logs for:
- Sample image loading at startup
- Cache point marker confirmation
- Bedrock response times
- Detection/tracking statistics

## Next Steps

1. Process test videos to verify end-to-end pipeline
2. Monitor Bedrock usage in AWS console
3. Verify prompt caching savings in billing
4. Adjust confidence thresholds if needed
5. Fine-tune tracking parameters for your use case
