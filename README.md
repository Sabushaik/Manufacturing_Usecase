# Manufacturing Use Case - PPE Detection API

This is a FastAPI application that provides PPE (Personal Protective Equipment) detection and safety analysis for manufacturing environments using YOLO detection + Amazon Bedrock Nova Lite Vision pipeline.

## Features

- **Video Upload**: Upload videos to S3 for processing
- **PPE Detection**: Detect hardhats, goggles, safety vests, gloves, and safety shoes
- **Vehicle Detection**: Identify industrial vehicles in the video
- **Person Tracking**: Track individuals throughout the video
- **Automated Report Generation**: Generate comprehensive PDF reports

## PDF Report Contents

The generated PDF report includes:

1. **Summary Section**
   - Presigned URLs for annotated video and results JSON
   - Number of persons detected in the video
   - Number of vehicles detected
   - Video duration and total frames processed
   - Safety violations summary (hardhat missing, shoes missing, gloves missing, vest missing, goggles missing)

2. **PPE Summary**
   - Count of each PPE item detected
   - Percentage of compliance for each PPE item
   - Table format for easy viewing

3. **Time Series Analysis**
   - Second-by-second breakdown of PPE violations
   - Shows violations for each PPE item across the video duration
   - Helps identify when violations occurred

4. **Graphical Analysis**
   - Line graph showing PPE violations over time
   - Visual representation of trends
   - Multiple lines for different PPE items

## Installation

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file with the following variables:

```
S3_BUCKET_NAME=your-input-bucket
OUTPUT_S3_BUCKET_NAME=your-output-bucket
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=us-east-1
OPENAI_KEY=your-openai-key
```

## Running the Application

```bash
python main.py
```

The server will start on `http://localhost:8092`

## API Endpoints

### 1. Health Check
```
GET /health
```

### 2. Upload Video
```
POST /upload-video
Content-Type: multipart/form-data

{
  "video": <video_file>
}
```

### 3. Process Video
```
POST /process-video
Content-Type: application/json

{
  "video_uri": "s3://bucket/key" or "https://presigned-url"
}
```

**Response includes:**
- `output_presigned_url`: URL to download the annotated video
- `results_json_presigned_url`: URL to download the detailed JSON results
- `report_pdf_presigned_url`: URL to download the PDF report
- `ppe_summary`: Aggregate PPE detection results
- `unique_counts`: Number of unique persons and vehicles detected

## Output Files

After processing, the following files are uploaded to S3:

1. **Annotated Video** (`*_annotated.mp4`): Original video with bounding boxes and PPE status overlays
2. **Results JSON** (`*_results.json`): Detailed JSON with frame-by-frame detection data
3. **PDF Report** (`*_report.pdf`): Comprehensive analysis report (NEW)

## Dependencies

- FastAPI: Web framework
- Triton Inference Server: YOLO model inference
- Amazon Bedrock: Vision analysis using Nova Lite model
- ReportLab: PDF generation
- Matplotlib: Graph generation
- OpenCV: Video processing
- Boto3: AWS S3 integration

## Report Generation Details

The PDF report generation happens automatically after video processing:

1. **Data Collection**: Aggregates detection results from the video processing pipeline
2. **Time Series Analysis**: Groups PPE violations by second for temporal analysis
3. **Graph Generation**: Creates matplotlib visualizations of violations over time
4. **PDF Assembly**: Uses ReportLab to create a professional PDF document
5. **S3 Upload**: Uploads the PDF to the output S3 bucket with a presigned URL

## Sample Output

The report includes:
- Executive summary with key metrics
- Detailed PPE compliance rates
- Time-series table showing violations per second
- Multi-line graph showing violation trends

## Notes

- Presigned URLs expire after 24 hours by default
- PDF reports are generated in memory to avoid disk space issues
- Graphs are temporarily saved and cleaned up after PDF generation
- The system handles failures gracefully - if PDF generation fails, the video and JSON results are still available
