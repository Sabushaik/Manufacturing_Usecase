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
The system now uses **two different counting methods**:

1. **Tracking IDs Count** (`tracking_ids_count`): 
   - Number of unique YOLO tracking IDs detected
   - Can be higher than actual persons due to re-assignments
   - Useful for debugging tracking issues

2. **Actual Person Count** (`unique_persons`):
   - Number of distinct persons verified by the **vision model** (Amazon Bedrock Nova Lite)
   - More accurate as it analyzes actual person appearance
   - This is now the **primary count** returned by the API

### Example Output
```
📊 YOLO Tracking IDs detected: 11
📊 Unique persons (from vision model): 5
📊 Unique vehicles: 2
```

In this example:
- YOLO assigned 11 different tracking IDs throughout the video
- The vision model confirmed only **5 distinct persons** actually exist
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

# NEW METHOD (accurate)
unique_persons = len(final_person_summary)  # Counts vision-verified persons
```

The `final_person_summary` dictionary contains one entry per **distinct person** detected by the vision model, with their PPE compliance status.

## Configuration
- **YOLO Model**: `yolo_person_detection`
- **Triton Server**: `localhost:8000`
- **Vision Model**: `amazon.nova-lite-v1:0`
- **Confidence Threshold**: 0.5
- **IOU Threshold**: 0.90
