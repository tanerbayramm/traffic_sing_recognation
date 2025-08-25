import cv2
from ultralytics import YOLO


model = YOLO("last.pt")
class_names = model.names


cap = cv2.VideoCapture(0)

last_class_name = None  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, save=False, imgsz=640, conf=0.8)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = class_names[class_id]
            # Inside your for box in boxes loop
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Expand the box slightly for better context (optional)
            pad = 10
            x1 = max(x1 - pad, 0)
            y1 = max(y1 - pad, 0)
            x2 = min(x2 + pad, frame.shape[1])
            y2 = min(y2 + pad, frame.shape[0])

            # Crop the region
            cropped = frame[y1:y2, x1:x2]

            # Resize (zoom)
            zoomed = cv2.resize(cropped, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

            # Optional: show or save zoomed
            cv2.imshow("Zoomed ROI", zoomed)

            # Continue drawing on original frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(frame, class_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            zoomed_results = model.predict(source=zoomed, save=False, imgsz=640, conf=0.5, iou=0.5)

            # Analyze new predictions
            for zr in zoomed_results:
                for zbox in zr.boxes:
                    zclass_id = int(zbox.cls[0])
                    zclass_name = class_names[zclass_id]

                    # Optional: update detection if zoomed prediction is different
                    if zclass_name != class_name:
                        print(f"Zoomed prediction updated class: {zclass_name}")
                        class_name = zclass_name  # replace original class name with more confident zoomed one

                        # Update text file if needed
                        if zclass_name != last_class_name:
                            last_class_name = zclass_name
                            with open("detected_class.txt", "w") as file:
                                confidence = float(box.conf[0])
                                file.write(f"{class_name} ({confidence:.2f})\n")

            
            if class_name != last_class_name:
                last_class_name = class_name
                with open("detected_class.txt", "w") as file:
                    confidence = float(box.conf[0])
                    file.write(f"{class_name} ({confidence:.2f})\n")
                

            
            

    cv2.imshow("Trafik İşareti Algılama", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
