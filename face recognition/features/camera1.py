import cv2
import numpy as np
from PIL import Image

from skimage.feature import Cascade
from torchvision.transforms import v2

from custom_dataset import val_tfms
import onnxruntime as ort


# --- 1. Load Model (ONNX) ---
# Oczekiwany wzorzec nazwy pliku: {FEATURE_NAME}_v{version}_best.onnx
MODEL_PATH = "./final_models/Male_v29_best.onnx"


def _create_session(model_path: str) -> ort.InferenceSession:
    # Preferuj CUDA jeśli dostępne, inaczej CPU.
    available = ort.get_available_providers()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    providers = [p for p in providers if p in available]
    return ort.InferenceSession(model_path, providers=providers)


session = _create_session(MODEL_PATH)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# --- 2. Setup Preprocessing ---
# To MUST match your Dataset transforms
transform = val_tfms


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


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


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

            # torch Tensor [C,H,W] float32
            input_tensor = transform(img_pil).unsqueeze(0)
            input_np = input_tensor.detach().cpu().numpy().astype(np.float32)

            logits = session.run([output_name], {input_name: input_np})[0]
            # logits shape: (1, 1)
            prob = float(_sigmoid(float(logits[0][0])))

            label = "Male" if prob > 0.5 else "Not Male"
            color = (0, 255, 0) if prob > 0.5 else (0, 0, 255)
            cv2.putText(frame, f"{label}: {prob:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


if __name__ == '__main__':
    file = "./face.xml"
    detector = Cascade(file)

    cap = cv2.VideoCapture(2)
    skip = 5
    i = 0
    boxes = []
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        if i % skip == 0:
            boxes = detect(frame, detector)

        draw(frame, boxes)
        cv2.imshow('Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i += 1

    cap.release()
    cv2.destroyAllWindows()
