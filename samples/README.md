# Sample Images for Prompt Caching

This directory should contain sample images for each PPE and vehicle category.
The images are used to provide visual guidance to the Bedrock Nova Lite model.

## Required Sample Images (12 total - 1 per category)

### PPE Categories (5 images):
- `hardhat/hardhat1.jpg` - Example of a hardhat/helmet
- `goggles/goggles1.jpg` - Example of safety goggles
- `vest/safety_vest1.jpg` - Example of a safety vest
- `gloves/gloves1.jpg` - Example of work gloves
- `shoes/shoes1.jpg` - Example of safety shoes

### Vehicle Categories (7 images):
- `forklift/forklift1.jpg` - Example of a forklift
- `truck/truck1.jpg` - Example of a truck
- `excavator/excavator1.jpg` - Example of an excavator
- `bull_dozer/bull_dozer1.jpg` - Example of a bulldozer
- `cement_mixer/cement_mixer1.jpg` - Example of a cement mixer
- `roller/roller1.jpg` - Example of a road roller
- `tracktor/tracktor1.jpg` - Example of a tractor (note: folder name is 'tracktor')

## Purpose

These sample images are:
1. Loaded once at application startup
2. Encoded as base64
3. Included in the system prompt sent to Bedrock Nova Lite
4. Marked with a `cachePoint` for prompt caching
5. Cached by Bedrock for efficient reuse across all requests

## Prompt Caching Benefits

By marking the system prompt + sample images with a cache point:
- **Startup**: Load 12 images → Build system message → Store in SYSTEM_MESSAGES global
- **Each Request**: Send cached system messages + new crop image
- **Cost Savings**: Bedrock caches the system prompt and sample images, reducing token costs
- **Performance**: Faster response times due to cached content

## Note

Place actual image files in the appropriate subdirectories before running the application.
If sample images are not found, the application will log warnings but continue to operate.
