# Manufacturing Use Case - PPE Detection API

## Overview
FastAPI application for PPE (Personal Protective Equipment) detection and vehicle identification using:
- **YOLO** (via Triton Inference Server) for object detection
- **BOTSORT** for multi-object tracking
- **Amazon Bedrock Nova Lite** for vision-based classification

## Person Counting Explanation

### The Issue
Previously, the system was showing inflated person counts (e.g., "11 persons" when only 5 were in the video).

### Why This Happened
1. **YOLO Detection**: Detects persons in each frame
2. **BOTSORT Tracking**: Assigns unique tracking IDs to track persons across frames
3. **Problem**: Tracking IDs can multiply when:
   - A person leaves and re-enters the frame
   - Occlusions cause the tracker to lose and reassign IDs
   - Camera angle changes or motion blur affects tracking
   - Multiple people overlap temporarily

### The Solution
The system now uses **vision model person counting with averaging**:

1. **Tracking IDs Count** (`tracking_ids_count`): 
   - Number of unique YOLO tracking IDs detected
   - Can be higher than actual persons due to re-assignments
   - Useful for debugging tracking issues

2. **Vision Model Person Count** (`unique_persons`):
   - The vision model (Amazon Bedrock Nova Lite) counts persons in **each crop/frame**
   - System calculates the **average** across all crops
   - **Example**: 120 crops analyzed
     - 40 crops report 4 persons
     - 80 crops report 5 persons
     - Average: (40×4 + 80×5) / 120 = 4.67 → **5 persons**
   - This is now the **primary count** returned by the API

### Example Output
```
📊 Vision model person counts: 120 crops analyzed
📊 Average person count: 4.67 → 5
📊 YOLO Tracking IDs detected: 11
📊 Unique persons (from vision model average): 5
📊 Unique vehicles: 2
```

In this example:
- Vision model analyzed 120 person crops from the video
- Average count across all crops: 4.67 → rounds to **5 persons**
- YOLO assigned 11 different tracking IDs (debug info)
- The API returns `unique_persons: 5` as the accurate count

## API Endpoints

### `/process-video`
Process a video with YOLO detection + Bedrock Nova Lite vision analysis.

**Response includes:**
```json
{
  "unique_counts": {
    "persons": 5,           // Actual distinct persons (from vision model)
    "tracking_ids": 11,     // YOLO tracking IDs (may be higher)
    "vehicles": 2
  },
  "per_person_ppe_summary": {
    "1": {
      "hardhat": true,
      "goggles": true,
      ...
    }
  }
}
```

## Technical Details

### Person Count Calculation
```python
# OLD METHOD (inaccurate)
unique_persons = len(person_active_ids)  # Counts YOLO tracking IDs

# NEW METHOD (accurate - averaging vision model counts)
person_counts_from_vision = []
for item in json_output:
    if item.get("gpt_result") and "person_count" in item["gpt_result"]:
        person_counts_from_vision.append(item["gpt_result"]["person_count"])

# Average and round to nearest integer
unique_persons = round(sum(person_counts_from_vision) / len(person_counts_from_vision))
```

The vision model returns `person_count` for each crop, indicating how many persons are visible in that specific image. The system averages all these counts to get the final person count.

## Configuration
- **YOLO Model**: `yolo_person_detection`
- **Triton Server**: `localhost:8000`
- **Vision Model**: `amazon.nova-lite-v1:0`
- **Confidence Threshold**: 0.5
- **IOU Threshold**: 0.90
