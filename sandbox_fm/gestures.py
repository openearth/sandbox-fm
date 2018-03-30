def recognize_gestures(height, lower_threshold=20):
    gestures = []
    if (height > lower_threshold).any():
        gestures.append({
            "name": "hand",
            "feature": height > lower_threshold
        })
    return gestures
