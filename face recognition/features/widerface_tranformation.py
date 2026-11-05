import os
from pathlib import Path

from PIL import Image
from tqdm import tqdm

WIDERFACE_DIR = Path(__file__).parent.parent / "data" / "widerface"
WIDERFACE_IMAGES_DIR = WIDERFACE_DIR / "WIDER_train" / "images"
# The format of txt ground truth.
# File name
# Number of bounding box
# x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose

def load_directories(widerface_path):
    return os.listdir(widerface_path)


def read_boxes(txt_path):
    boxes = {}
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.endswith('.jpg'):
                image_name = line
                i += 1
                num_boxes = int(lines[i].strip())
                i += 1
                if num_boxes >= 3 and num_boxes <= 5:
                    boxes[image_name] = []
                    for _ in range(num_boxes):
                        box_info = list(map(int, lines[i].strip().split()))
                        boxes[image_name].append(box_info)
                        i += 1
                else:
                    i += num_boxes
            else:
                i += 1
    return boxes

def crop_and_save(image_path, box, output_dir, name):
    target_size = (178, 218)
    image = Image.open(image_path)
    x, y, w, h, _, _, _, _, _, _ = box
    if w <= 0 or h <= 0:
        print(f"Invalid box {box} for image {image_path}")
        return
    img_w, img_h = image.size
    if not (80 <= h <= 140):
        print(f"Bounding box too small {(x+w, y+h)} for image {image_path} box {box}")
        return
    center_x = x + (w / 2)
    center_y = y + (h / 2)
    left = center_x - (target_size[0] / 2)
    right = left + target_size[0]
    top = center_y - (target_size[1] / 2)
    bottom = top + target_size[1]
    if left < 0:
        right -= left  # Move right by the amount we were over the left edge
        left = 0
    if right > img_w:
        left -= (right - img_w)  # Move left by the amount we were over the right edge
        right = img_w
        # Shift Vertical
    if top < 0:
        bottom -= top
        top = 0
    if bottom > img_h:
        top -= (bottom - img_h)
        bottom = img_h
    cropped_image = image.crop((int(left), int(top), int(right), int(bottom)))
    if cropped_image.size[0] < 100 or cropped_image.size[1] < 100:
        return
    cropped_image.save(output_dir / name)

def crop_and_save_all(widerface_path, output_dir):
    boxes = read_boxes(WIDERFACE_DIR / 'wider_face_train_bbx_gt.txt')
    for image_path in tqdm(boxes.keys(), desc="Processing images", total=len(boxes)):
        name = image_path.split("/")[-1] if image_path.find("/") != -1 else image_path.split("\\")[-1]
        name = name.split(".")[0] + "_"
        for index, box in enumerate(boxes[image_path]):
            try:
                crop_and_save(widerface_path / image_path, box, output_dir, name + str(index) + ".jpg")
            except Exception as e:
                print(f"Error processing {image_path} box {index} with values: {box}, name: {name}")
                print(e)
                continue

if __name__ == '__main__':
    os.makedirs(WIDERFACE_DIR / 'transformed_padding', exist_ok=True)
    crop_and_save_all(WIDERFACE_IMAGES_DIR, WIDERFACE_DIR / 'transformed_padding')


