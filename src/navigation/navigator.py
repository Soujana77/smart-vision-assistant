def analyze_path(detections):
    """
    Analyze detected objects and determine
    whether the path is safe.
    """

    important_objects = [
        "person",
        "chair",
        "car",
        "truck",
        "bus",
        "motorcycle",
        "bicycle"
    ]

    for detection in detections:

        class_name = detection["class_name"]
        direction = detection["direction"]
        distance = detection["distance"]

        if (
            class_name in important_objects
            and direction == "ahead"
            and distance in ["close", "very close"]
        ):

            return "Obstacle ahead. Move carefully."

    return "Path appears clear."