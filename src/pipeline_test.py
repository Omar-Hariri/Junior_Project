import cv2
import winsound
from core.pipeline import process_frame

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    
    results = process_frame(frame)
    
    if results:
        l_state = results["left"]["state"]
        r_state = results["right"]["state"]


        for side, data in results.items():
            x1, y1, x2, y2 = data["bbox"]
            color = (0, 0, 255) if data["state"] == 1 else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


        if l_state == 1 and r_state == 1:
            cv2.putText(frame, "DROWSY ALERT!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            winsound.Beep(2000, 500)
            
    cv2.imshow("Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()