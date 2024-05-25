import os
import cv2
import numpy as np
from ultralytics import YOLO  # Ensure this import is correct for your setup
from skimage import feature, morphology
from skimage.morphology import dilation, erosion

# Initialize the YOLO model
model = YOLO("yolov8x.pt")  # Adjust the model path and version as necessary


def resize_image(image, target_size=640):
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    final_image = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    top, left = (target_size - new_h) // 2, (target_size - new_w) // 2
    final_image[top:top + new_h, left:left + new_w] = resized_image
    return final_image


def intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    edges = feature.canny(clahe_img, sigma=1, low_threshold=10, high_threshold=50)
    edges = edges.astype(np.uint8) * 255  # Convert edges to uint8 type
    dilated = dilation(edges, morphology.square(3))
    eroded = erosion(dilated, morphology.square(3))
    return eroded


def find_contours(image):
    processed = preprocess_image(image)
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def combine_boxes(boxA, boxB):
    x1 = min(boxA[0], boxB[0])
    y1 = min(boxA[1], boxB[1])
    x2 = max(boxA[2], boxB[2])
    y2 = max(boxA[3], boxB[3])
    return x1, y1, x2, y2


def grabcut_segmentation(image, rect):
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    segmented_image = image * mask2[:, :, np.newaxis]
    return segmented_image


def graham_scan(points):
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    points = sorted(points)
    if len(points) <= 1:
        return points

    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def process_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = resize_image(img)
    results = model.predict(img, classes=[0, 1, 3])  # 0=person, 1=bicycle, 3=motorcycle
    overlay = img.copy()

    person_boxes = []
    ride_boxes = []

    for result in results:
        for det in result.boxes:
            bbox = det.xyxy[0].cpu().numpy()
            if int(det.cls) == 0:
                person_boxes.append(bbox)
            elif int(det.cls) in [1, 3]:
                ride_boxes.append(bbox)

    for person in person_boxes:
        for ride in ride_boxes:
            if intersection_over_union(person, ride) > 0.1:
                combined_box = combine_boxes(person, ride)
                x1, y1, x2, y2 = map(int, combined_box)

                # Apply GrabCut for background removal
                rect = (x1, y1, x2 - x1, y2 - y1)
                segmented_img = grabcut_segmentation(img, rect)
                crop_segmented = segmented_img[y1:y2, x1:x2]

                contours = find_contours(crop_segmented)
                hull_points = []
                for contour in contours:
                    if cv2.contourArea(contour) > 50:  # Increased the threshold to reduce noise
                        mapped_contour = contour + np.array([[x1, y1]])
                        hull_points.extend(mapped_contour)
                if hull_points:
                    hull_points = np.array(hull_points).reshape(-1, 2)
                    hull = graham_scan(hull_points.tolist())
                    for i in range(len(hull)):
                        pt1 = tuple(hull[i])
                        pt2 = tuple(hull[(i + 1) % len(hull)])
                        cv2.line(overlay, pt1, pt2, (0, 255, 0), 2)  # Draw line with green color

    final_image = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
    return final_image


def load_images_from_folder(folder):
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg')):
            img_path = os.path.join(folder, filename)
            processed_img = process_image(img_path)
            if processed_img is not None:
                cv2.imshow("Processed Image", processed_img)
                cv2.waitKey(0)  # Wait for a key press to show the next image
    cv2.destroyAllWindows()


folder_path = r'C:\University\2. letnik\Uvod v racunalnisko geometrijo\Projektna2\Slike'
load_images_from_folder(folder_path)
