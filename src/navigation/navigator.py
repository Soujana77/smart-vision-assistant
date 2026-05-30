def analyze_path(detections):

    important_objects = [
        "person",
        "chair",
        "car",
        "truck",
        "bus",
        "motorcycle",
        "bicycle",
        "table"
    ]

    left_blocked = False
    center_blocked = False
    right_blocked = False

    for detection in detections:

        class_name = detection["class_name"]
        direction = detection["direction"]
        distance = detection["distance"]

        if (
            class_name in important_objects
            and distance in ["close", "very close"]
        ):

            if direction == "on the left":
                left_blocked = True

            elif direction == "ahead":
                center_blocked = True

            elif direction == "on the right":
                right_blocked = True

    # Decision Engine

    if center_blocked:

        if not left_blocked:
            return "Obstacle ahead. Move left."

        elif not right_blocked:
            return "Obstacle ahead. Move right."

        else:
            return "Stop. Path blocked."

    if left_blocked and not right_blocked:
        return "Obstacle on left. Keep right."

    if right_blocked and not left_blocked:
        return "Obstacle on right. Keep left."

    return "Path clear. Move forward."