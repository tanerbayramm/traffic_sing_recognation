import cv2
from ultralytics import YOLO


model = YOLO("best.pt")
class_names = model.names


cap = cv2.VideoCapture(0)

last_class_name = None  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, save=False, imgsz=640, conf=0.5)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = class_names[class_id]

            
            if class_name != last_class_name:
                last_class_name = class_name
                with open("detected_class.txt", "w") as file:
                    file.write(f"{class_name}\n")
                print(f"Yeni sınıf algılandı: {class_name}")

            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(frame, class_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

    cv2.imshow("Trafik İşareti Algılama", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
