import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model and labels
model = load_model("model/hand_model.h5")
label_names = np.load("labels.npy")

IMG_SIZE = 64

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # ROI box
    x1, y1, x2, y2 = 50, 50, 350, 350
    roi = frame[y1:y2, x1:x2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    normalized = resized / 255.0

    reshaped = normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1).astype('float32')

    prediction = model.predict(reshaped, verbose=0)
    class_id = np.argmax(prediction)
    label = label_names[class_id]

    # Draw box + label
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(frame, f"Sign: {label}", (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Hand Sign Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()