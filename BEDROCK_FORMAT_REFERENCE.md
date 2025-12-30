# Bedrock Nova Lite API Format Reference

## Issue Resolution

**Problem**: ValidationException - "extraneous key [type] is not permitted"

**Root Cause**: The Bedrock Nova API has a specific format for system content blocks that differs from standard message content blocks.

**Fixed in**: Commit 8ee6a4e

## Correct Format for System Content Blocks

### ✅ Correct Format

```python
system_messages = [
    # Text block
    {"text": "You are a vision analysis system..."},
    
    # Text label
    {"text": "Example: hardhat"},
    
    # Image block
    {
        "image": {
            "format": "jpeg",
            "source": {
                "bytes": "base64_encoded_string"
            }
        }
    },
    
    # Last block with cache point
    {
        "image": {
            "format": "jpeg",
            "source": {
                "bytes": "base64_encoded_string"
            }
        },
        "cachePoint": {"type": "default"}
    }
]
```

### ❌ Incorrect Format (Causes ValidationException)

```python
# DO NOT USE - This will fail
system_messages = [
    {"type": "text", "text": "..."},  # ❌ 'type' key not allowed
    {"type": "image", "source": {...}}  # ❌ 'type' key not allowed
]
```

## Key Differences

| Aspect | System Blocks | User Message Blocks |
|--------|--------------|---------------------|
| Root keys | `text` or `image` | Can use `type` wrapper |
| Image format | `{"image": {"format": "jpeg", ...}}` | `{"image": {"format": "jpeg", ...}}` |
| Type key | ❌ NOT allowed | ✅ Optional |
| Cache point | ✅ Allowed on last block | ❌ Not applicable |

## Complete Payload Structure

```python
payload = {
    # System messages (cached)
    "system": [
        {"text": "System prompt..."},
        {"text": "Example: category1"},
        {"image": {"format": "jpeg", "source": {"bytes": "..."}}},
        # ... more examples ...
        {"image": {...}, "cachePoint": {"type": "default"}}
    ],
    
    # User messages (new each time)
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "image": {
                        "format": "jpeg",
                        "source": {"bytes": "crop_image_base64"}
                    }
                },
                {
                    "text": "Analyze this image..."
                }
            ]
        }
    ]
}
```

## Cache Point Format

The `cachePoint` marker tells Bedrock to cache content up to and including this block:

```python
# Correct cache point format
{"cachePoint": {"type": "default"}}

# This is applied to the LAST content block in the system messages
```

## Implementation in Code

### build_system_messages Function

```python
def build_system_messages(sample_cache: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """Build system messages with correct Bedrock format"""
    msgs = [{"text": SYSTEM_TEXT}]  # ✅ No 'type' key
    
    for label, encoded_images in sample_cache.items():
        for img_b64 in encoded_images: 
            msgs.append({"text": f"Example: {label}"})  # ✅ No 'type' key
            msgs.append({
                "image": {  # ✅ 'image' at root, not 'type'
                    "format": "jpeg",
                    "source": {
                        "bytes": img_b64
                    }
                }
            })
    
    # Add cache point to last block
    if msgs:
        msgs[-1]["cachePoint"] = {"type": "default"}
    
    return msgs
```

## Testing

Verify the format is correct:

```bash
python test_caching_structure.py
```

Expected output:
```
✅ SAMPLES has 12 categories with 12 total images
✅ System messages structure correct:
   - 13 text blocks (1 system + 12 labels)
   - 12 image blocks
   - cachePoint marker on last block: {'type': 'default'}
✅ Payload structure correct
✅ All categories present
```

## Common Errors and Solutions

### Error 1: "extraneous key [type] is not permitted"
**Solution**: Remove `"type"` key from system content blocks. Use `"text"` or `"image"` directly at root level.

### Error 2: "Invalid image format"
**Solution**: Ensure image blocks have this structure:
```python
{"image": {"format": "jpeg", "source": {"bytes": "..."}}}
```

### Error 3: "cachePoint not recognized"
**Solution**: Use correct format:
```python
{"cachePoint": {"type": "default"}}
```

## References

- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Bedrock Nova Models](https://docs.aws.amazon.com/bedrock/latest/userguide/nova-models.html)
- [Prompt Caching Guide](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html)

## Summary

The key takeaway: **Bedrock system content blocks use a simplified format without the `type` wrapper**. Always use `text` or `image` directly as root keys, not as values in a `type` field.
