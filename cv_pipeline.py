import cv2
import time
import csv
from ultralytics import YOLO

# ==============================
# Load YOLO model
# ==============================
model = YOLO("yolo11n.pt")

# ==============================
# Open input video
# ==============================
cap = cv2.VideoCapture("input.mp4")

if not cap.isOpened():
    print("Error: Could not open input.mp4")
    exit()

# ==============================
# Video properties
# ==============================
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_input = cap.get(cv2.CAP_PROP_FPS)

# ==============================
# Output video
# ==============================
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    "output_annotated.mp4",
    fourcc,
    fps_input,
    (width, height)
)

# ==============================
# CSV file
# ==============================
csv_file = open(
    "latency.csv",
    "w",
    newline=""
)

csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "frame",
    "latency_ms",
    "fps"
])

frame_number = 0

print("Processing video...")
print("Press Q to stop early.")

# ==============================
# Process video
# ==============================
while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # ==============================
    # Start latency timer
    # ==============================
    start_time = time.perf_counter()

    # ==============================
    # YOLO tracking
    # ==============================
    results = model.track(
        frame,
        classes=[0],
        persist=True,
        verbose=False
    )

    # ==============================
    # End timer
    # ==============================
    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000

    fps = 1000 / latency_ms if latency_ms > 0 else 0

    # ==============================
    # Draw detections
    # ==============================
    annotated_frame = results[0].plot()

    # ==============================
    # Tracking information
    # ==============================
    if results[0].boxes is not None:

        boxes = results[0].boxes

        if boxes.id is not None:

            track_ids = boxes.id.int().cpu().tolist()

            coordinates = boxes.xyxy.int().cpu().tolist()

            for box, track_id in zip(
                coordinates,
                track_ids
            ):

                x1, y1, x2, y2 = box

                # ==============================
                # Centroid
                # ==============================
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # ==============================
                # Bounding box size
                # ==============================
                box_width = x2 - x1
                box_height = y2 - y1

                box_area = box_width * box_height

                # ==============================
                # Draw centroid
                # ==============================
                cv2.circle(
                    annotated_frame,
                    (cx, cy),
                    5,
                    (0, 0, 255),
                    -1
                )

                # ==============================
                # Display information
                # ==============================
                text = (
                    f"ID:{track_id} "
                    f"C:({cx},{cy}) "
                    f"Area:{box_area}"
                )

                cv2.putText(
                    annotated_frame,
                    text,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )

    # ==============================
    # Display latency
    # ==============================
    cv2.putText(
        annotated_frame,
        f"Latency: {latency_ms:.1f} ms",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # ==============================
    # Display FPS
    # ==============================
    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # ==============================
    # Save annotated frame
    # ==============================
    out.write(annotated_frame)

    # ==============================
    # Save latency information
    # ==============================
    csv_writer.writerow([
        frame_number,
        round(latency_ms, 2),
        round(fps, 2)
    ])

    # ==============================
    # Show video
    # ==============================
    cv2.imshow(
        "YOLO Tracking + Centroid",
        annotated_frame
    )

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==============================
# Release everything
# ==============================
cap.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()

print()
print("================================")
print("Processing completed!")
print("================================")
print(f"Total frames processed: {frame_number}")
print("Output video: output_annotated.mp4")
print("Latency file: latency.csv")
