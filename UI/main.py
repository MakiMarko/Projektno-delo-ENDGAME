from tkinter import Tk, Button, Label, filedialog
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import pygame

# Initialize the YOLO model
model = YOLO("yolov8n.pt")

pygame.mixer.init()
# Sound effects paths
sound_effect_1 = pygame.mixer.Sound('./sound effects/Danger Alarm Sound Effect.mp3')
sound_effect_2 = pygame.mixer.Sound('./sound effects/Tom Screaming Sound Effect (From Tom and Jerry).mp3')

crossed = {}

# Initialize counters for consecutive frames with no detection
no_detection_count = {0: 0, 1: 0, 2: 0}
# Threshold for consecutive frames without detection to update message
no_detection_threshold = 5  # Adjust this threshold as needed

# Initialize the GUI
root = Tk()
root.title("Video Loader")
root.geometry('1200x500')

# Label to display the video frames
label_video1 = Label(root)
label_video1.grid(row=0, column=0)
label_video2 = Label(root)
label_video2.grid(row=0, column=1)
label_video3 = Label(root)
label_video3.grid(row=0, column=2)

# Labels for status messages under each video
label_message1 = Label(root, text="")
label_message1.grid(row=1, column=0)
label_message2 = Label(root, text="")
label_message2.grid(row=1, column=1)
label_message3 = Label(root, text="")
label_message3.grid(row=1, column=2)

caps = [None, None, None]


def play_sound(sound):
    pygame.mixer.Sound.play(sound)


def update_message(msg, index):
    if index == 0:
        label_message1.config(text=msg)
    elif index == 1:
        label_message2.config(text=msg)
    elif index == 2:
        label_message3.config(text=msg)


def combine_boxes(boxA, boxB):
    x1 = min(boxA[0], boxB[0])
    y1 = min(boxA[1], boxB[1])
    x2 = max(boxA[2], boxB[2])
    y2 = max(boxA[3], boxB[3])
    return x1, y1, x2, y2


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


def check_and_update_status_for_sides(line_position, y2, box_id, index):
    global crossed
    if y2 > line_position:
        if not crossed[box_id]['line1']:
            play_sound(sound_effect_2)
            update_message("Danger", index)
            crossed[box_id]['line1'] = True
    else:
        update_message("Safe distance", index)


def check_and_update_status_for_middle(x1, x2, box_id, line_position_1, line_position_2, index):
    if x1 < line_position_1:
        zone = "Zone 1"
    elif x1 > line_position_1 and x2 < line_position_2:
        zone = "Zone 2"
    else:
        zone = "Zone 3"
    update_message(f"Biker in {zone}", index)


def process_frame(cap, index):
    global crossed, no_detection_count
    if cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cropped_height = int(frame.shape[0] * 0.9)
            frame = frame[:cropped_height, :]
            frame = cv2.resize(frame, (400, 280))
            height = frame.shape[0]
            overlay = frame.copy()
            line_color = (0, 0, 0)
            line_thickness = 2
            alpha = 0.3

            line_position = None
            line_position_1 = None
            line_position_2 = None

            if index == 1:  # Middle video
                line_position_1 = int(frame.shape[1] * 0.33)
                line_position_2 = int(frame.shape[1] * 0.66)
                cv2.line(overlay, (line_position_1, 0), (line_position_1, height), line_color, line_thickness)
                cv2.line(overlay, (line_position_2, 0), (line_position_2, height), line_color, line_thickness)
            else:  # Other videos
                line_position = int(height * 0.85)
                cv2.line(overlay, (0, line_position), (frame.shape[1], line_position), line_color, line_thickness)

            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            results = model.predict(frame, classes=[0, 1, 3])
            person_boxes = []
            ride_boxes = []

            detected = False  # Flag to check if any biker is detected

            for result in results:
                for det in result.boxes:
                    bbox = det.xyxy[0].cpu().numpy()
                    if int(det.cls) == 0:
                        person_boxes.append(bbox)
                        detected = True
                    elif int(det.cls) in [1, 3]:
                        ride_boxes.append(bbox)

            if detected:
                no_detection_count[index] = 0  # Reset the counter on detection
                merged_boxes = []

                for person in person_boxes:
                    for ride in ride_boxes:
                        if intersection_over_union(person, ride) > 0.1:
                            combined_box = combine_boxes(person, ride)
                            merged_boxes.append(combined_box)

                for box in merged_boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    box_id = id(box)

                    if box_id not in crossed:
                        crossed[box_id] = {'line1': False, 'line2': False}

                    if index == 1:
                        check_and_update_status_for_middle(x1, x2, box_id, line_position_1, line_position_2, index)
                    else:
                        check_and_update_status_for_sides(line_position, y2, box_id, index)

            else:
                no_detection_count[index] += 1
                if no_detection_count[index] >= no_detection_threshold:
                    update_message("No cyclist/biker detected.", index)

            return frame
    return None


def update_frames():
    frames = [process_frame(cap, i) for i, cap in enumerate(caps)]
    labels = [label_video1, label_video2, label_video3]
    for frame, label in zip(frames, labels):
        if frame is not None:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            photo = ImageTk.PhotoImage(image=image)
            label.config(image=photo)
            label.image = photo
    root.after(25, update_frames)


def load_video(index):
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    if video_path:
        if caps[index] is not None:
            caps[index].release()
        caps[index] = cv2.VideoCapture(video_path)


button_load1 = Button(root, text="Load Video 1", command=lambda: load_video(0))
button_load1.grid(row=2, column=0, pady=20)
button_load2 = Button(root, text="Load Video 2", command=lambda: load_video(1))
button_load2.grid(row=2, column=1, pady=20)
button_load3 = Button(root, text="Load Video 3", command=lambda: load_video(2))
button_load3.grid(row=2, column=2, pady=20)

button_start = Button(root, text="Start Videos", command=update_frames)
button_start.grid(row=3, column=1, pady=20)

root.mainloop()
