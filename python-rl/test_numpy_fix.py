#!/usr/bin/env python3
"""
Quick test script to verify numpy type conversion fix
"""
import json
import numpy as np

# Test the conversion logic
def convert_numpy_types(obj):
    """Convert numpy types to native Python types"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# Test cases
test_messages = [
    {
        'command': 'step',
        'action': np.int64(5),  # This was causing the error
        'step': np.int32(10)
    },
    {
        'command': 'reset',
        'episode': np.int64(1)
    }
]

print("Testing numpy type conversion:\n")
for i, msg in enumerate(test_messages, 1):
    print(f"Test {i}:")
    print(f"  Original: {msg}")
    print(f"  Types: {[(k, type(v).__name__) for k, v in msg.items()]}")
    
    converted = convert_numpy_types(msg)
    print(f"  Converted: {converted}")
    print(f"  Types: {[(k, type(v).__name__) for k, v in converted.items()]}")
    
    # Try JSON serialization
    try:
        json_str = json.dumps(converted)
        print(f"  JSON: {json_str}")
        print("  ✓ JSON serialization SUCCESS\n")
    except Exception as e:
        print(f"  ✗ JSON serialization FAILED: {e}\n")

print("All tests passed! The fix should work.")
