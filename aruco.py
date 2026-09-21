import cv2
import json

# Define the dictionary mapping ArUco IDs to Hex Types
HEX_TYPE_MAP = {
    range(0, 10): "forest",
    range(10, 20): "pasture",
    range(20, 30): "field",
    range(30, 40): "hill",
    range(40, 50): "mountain",
    range(50, 60): "desert"
}

def get_hex_type(marker_id):
    """Maps a marker ID to its corresponding hex resource type."""
    for id_range, hex_type in HEX_TYPE_MAP.items():
        if marker_id in id_range:
            return hex_type
    return "unknown"

# Initialize OpenCV ArUco detector (DICT_4X4_50 or DICT_6X6_250 are good options)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for optimal marker detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect markers
    corners, ids, rejected = detector.detectMarkers(gray)

    board_state = []

    if ids is not None:
        # Flatten IDs array
        ids = ids.flatten()

        for marker_corners, marker_id in zip(corners, ids):
            # Extract corner points (shape: 1x4x2)
            pts = marker_corners.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = pts

            # Calculate centroid of the marker (which is the center of the hex tile)
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)

            hex_type = get_hex_type(marker_id)

            board_state.append({
                "marker_id": int(marker_id),
                "type": hex_type,
                "center": {"x": cX, "y": cY},
                "corners": pts.tolist()
            })

            # Draw visual feedback on frame
            cv2.polylines(frame, [pts.astype(int)], True, (0, 255, 0), 2)
            cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"{hex_type} ({marker_id})", (cX - 30, cY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Clean output JSON representation of the current layout
    json_output = json.dumps({"hex_count": len(board_state), "hexes": board_state}, indent=2)

    cv2.imshow("ArUco Catan Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
