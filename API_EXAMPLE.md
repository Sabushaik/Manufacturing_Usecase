# API Response Example

## /process-video Response

When you call the `/process-video` endpoint, you will receive a response that now includes the PDF report:

```json
{
  "message": "Successfully processed video with YOLO + Bedrock Nova Lite Vision pipeline",
  "input_uri": "s3://your-bucket/videos/20240107_120000_sample.mp4",
  "input_type": "s3_uri",
  "output_s3_uri": "s3://your-output-bucket/videos/20240107_120000_sample_annotated.mp4",
  "output_presigned_url": "https://your-output-bucket.s3.amazonaws.com/videos/20240107_120000_sample_annotated.mp4?...",
  "results_json_uri": "s3://your-output-bucket/videos/20240107_120000_sample_results.json",
  "results_json_presigned_url": "https://your-output-bucket.s3.amazonaws.com/videos/20240107_120000_sample_results.json?...",
  "report_pdf_uri": "s3://your-output-bucket/videos/20240107_120000_sample_report.pdf",
  "report_pdf_presigned_url": "https://your-output-bucket.s3.amazonaws.com/videos/20240107_120000_sample_report.pdf?...",
  "presigned_url_expiration": "24 hours",
  "config": {
    "model_name": "yolo_person_detection",
    "triton_server": "localhost:8000",
    "vision_model": "amazon.nova-lite-v1:0",
    "vision_provider": "Amazon Bedrock",
    "confidence_threshold": 0.65,
    "iou_threshold": 0.9
  },
  "unique_counts": {
    "persons": 3,
    "vehicles": 2
  },
  "per_person_ppe_summary": {
    "1": {
      "hardhat": true,
      "goggles": true,
      "safety_vest": true,
      "gloves": true,
      "shoes": true,
      "PPE": true
    },
    "2": {
      "hardhat": false,
      "goggles": true,
      "safety_vest": true,
      "gloves": false,
      "shoes": true,
      "PPE": false
    }
  },
  "ppe_summary": {
    "hard_hat": 2,
    "goggles": 3,
    "safety_vest": 3,
    "gloves": 2,
    "safety_shoes": 3,
    "persons_detected": 3,
    "list_of_vehicles": ["Forklift", "Truck"]
  },
  "vehicle_counts": {
    "Forklift": 1,
    "Truck": 1
  },
  "video_metadata": {
    "width": 1920,
    "height": 1080,
    "fps": 30,
    "duration": 8.5
  },
  "timing": {
    "api_invoked_at": "2024-01-07T12:00:00",
    "s3_download_duration": 2.5,
    "processing_duration": 45.3,
    "video_conversion_duration": 3.2,
    "s3_upload_video_duration": 4.1,
    "s3_upload_json_duration": 0.3,
    "pdf_generation_duration": 1.8,
    "s3_upload_pdf_duration": 0.5,
    "total_duration": 57.7
  },
  "detailed_processing_timings": {
    "detection_tracking_total": 30.2,
    "triton_infer_total": 15.5,
    "postprocess_total": 2.3,
    "tracking_total": 3.1,
    "gpt_total": 8.9,
    "annotate_total": 4.2,
    "total": 45.3
  }
}
```

## New Fields in Response

The following fields have been added to support PDF report generation:

1. **`report_pdf_uri`**: S3 URI of the generated PDF report
2. **`report_pdf_presigned_url`**: Presigned URL to download the PDF report (expires in 24 hours)
3. **`timing.pdf_generation_duration`**: Time taken to generate the PDF (in seconds)
4. **`timing.s3_upload_pdf_duration`**: Time taken to upload the PDF to S3 (in seconds)

## PDF Report Content Preview

The PDF report includes:

### 1. Summary Section
- Clickable links to annotated video and results JSON
- Number of persons detected: 3
- Number of vehicles detected: 2
- Video duration: 8.5 seconds
- Safety violations:
  - Hardhat Missing: 1 person(s)
  - Gloves Missing: 1 person(s)

### 2. PPE Summary Table
| PPE Item | Count Detected | Percentage |
|----------|----------------|------------|
| Hard Hat | 2 | 66.7% |
| Goggles | 3 | 100.0% |
| Safety Vest | 3 | 100.0% |
| Gloves | 2 | 66.7% |
| Safety Shoes | 3 | 100.0% |

### 3. Time Series Analysis Table
| Time (sec) | Hardhat | Goggles | Vest | Gloves | Shoes |
|------------|---------|---------|------|--------|-------|
| 0-1 | 1 | 0 | 0 | 1 | 0 |
| 1-2 | 1 | 0 | 0 | 1 | 0 |
| 2-3 | 1 | 0 | 0 | 1 | 0 |
| ... | ... | ... | ... | ... | ... |

### 4. Graph - PPE Violations Over Time
A multi-line graph showing:
- X-axis: Time in seconds
- Y-axis: Number of violations
- Lines for each PPE item (Hardhat, Goggles, Safety Vest, Gloves, Safety Shoes)

## Usage Example

```python
import requests

# Process video
response = requests.post(
    "http://localhost:8092/process-video",
    json={"video_uri": "s3://bucket/video.mp4"}
)

result = response.json()

# Download the PDF report
pdf_url = result["report_pdf_presigned_url"]
pdf_response = requests.get(pdf_url)

with open("report.pdf", "wb") as f:
    f.write(pdf_response.content)

print(f"PDF report downloaded: report.pdf")
```

## Notes

- The PDF is generated automatically after video processing completes
- If PDF generation fails, the video and JSON results are still available
- All presigned URLs expire after 24 hours
- The PDF report is self-contained and includes all analysis results
