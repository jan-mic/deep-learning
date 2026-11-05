import cv2
import torch
from PIL import Image

from skimage.data import lbp_frontal_face_cascade_filename
from skimage.feature import Cascade
from torchvision.transforms import v2

from custom_model import CustomModel

# --- 1. Load Model ---
MODEL_PATH = "./lightning_logs/tensorboard_logs/version_24/checkpoints/best-loss-epoch=09-val_loss=0.05.ckpt"
model = CustomModel.load_from_checkpoint(MODEL_PATH)
model.eval()
model.freeze()

# --- 2. Setup Preprocessing ---
# This MUST match your Dataset transforms
transform = v2.Compose([
    v2.Pad(padding=(20, 0), fill=0),
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def detect(frame, detector):
    detections = detector.detect_multi_scale(img=frame, scale_factor=1.2, step_ratio=1,
                                             min_size=(100, 100), max_size=(200, 200))
    boxes = []
    for detection in detections:
        x = detection['c']
        y = detection['r']
        w = detection['width']
        h = detection['height']
        boxes.append((x, y, w, h))

    return boxes


def draw(frame, boxes):
    for x, y, w, h in boxes:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2)

        cx, cy = x + w // 2, y + h // 2
        left, top = int(cx - 178/2), int(cy - 218/2)
        right, bottom = left + 178, top + 218

        left, top = max(0, left), max(0, top)
        right, bottom = min(frame.shape[1], right), min(frame.shape[0], bottom)

        face_crop = frame[top:bottom, left:right]
        if not (face_crop.shape[0] == 0 or face_crop.shape[1] == 0):
            img_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))

            input_tensor = transform(img_pil).unsqueeze(0)  # Add batch dimension
            input_tensor = input_tensor.to(model.device)

            with torch.no_grad():
                logits = model(input_tensor)
                prob = torch.sigmoid(logits).item()

            label = "Glasses" if prob > 0.5 else "No Glasses"  # Example for 'Male' attribute
            color = (0, 255, 0) if prob > 0.5 else (0, 0, 255)
            cv2.putText(frame, f"{label}: {prob:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


if __name__ == '__main__':
    # file = lbp_frontal_face_cascade_filename()
    file = "./face.xml"
    detector = Cascade(file)

    cap = cv2.VideoCapture(2)
    skip = 5
    i = 0
    boxes = []
    while (True):
        ret, frame = cap.read()
        if i % skip == 0:
            boxes = detect(frame, detector)

        draw(frame, boxes)
        cv2.imshow('Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i += 1
    # cap.release()
    # cv2.destroyAllWindows()
