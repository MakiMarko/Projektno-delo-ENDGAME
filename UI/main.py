from tkinter import Tk, ttk, Button, Label, filedialog
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import pygame
from utils import combine_boxes, intersection_over_union

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

style = ttk.Style()
style.configure('TButton', paddiong=6, relief="flat", background="#ccc", font=('Helvetica', 12))
style.map('TButton',
          foreground=[('pressed', 'blue'), ('active', 'blue')],
          background=[('pressed', '!disabled', 'black'), ('active', 'white')])

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

def load_image(file_path):
    """Load an image from the disk and convert it to a Tkinter compatible format."""
    image = Image.open(file_path)
    image = image.resize((35, 35))
    return ImageTk.PhotoImage(image)

def play_sound(sound):
    pygame.mixer.Sound.play(sound, maxtime=2000, loops=0)


def update_message(msg, index):
    if index == 0:
        label_message1.config(text=msg)
    elif index == 1:
        label_message2.config(text=msg)
    elif index == 2:
        label_message3.config(text=msg)

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
            height, width = frame.shape[0], frame.shape[1]
            overlay = frame.copy()
            red_overlay = frame.copy()
            line_color = (0, 0, 0)
            line_thickness = 2
            alpha = 0.3
            red_alpha = 0.2  # Transparency factor for red overlay

            line_position_horizontal = int(height * 0.75)
            line_position_1 = int(width * 0.33)
            line_position_2 = int(width * 0.66)
            if index == 1:  # Only for the middle video
                cv2.line(overlay, (line_position_1, 0), (line_position_1, height), line_color, line_thickness)
                cv2.line(overlay, (line_position_2, 0), (line_position_2, height), line_color, line_thickness)
            else:
                cv2.line(overlay, (0, line_position_horizontal), (width, line_position_horizontal), line_color,
                         line_thickness)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            results = model.predict(frame, classes=[0, 1, 3])
            person_boxes = []
            bike_boxes = []
            bicycle_boxes = []
            zones_detected = [False, False, False]
            message_parts = ["", "", ""]

            for result in results:
                for det in result.boxes:
                    bbox = det.xyxy[0].cpu().numpy()
                    if int(det.cls) == 0:
                        person_boxes.append(bbox)
                    elif int(det.cls) in [1, 3]:
                        if int(det.cls) == 1:
                            bicycle_boxes.append(bbox)
                        elif int(det.cls) == 3:
                            bike_boxes.append(bbox)

            merged_boxes = []
            for person in person_boxes:
                for bicycle in bicycle_boxes:
                    if intersection_over_union(person, bicycle) > 0.1:
                        combined_box = combine_boxes(person, bicycle)
                        merged_boxes.append(combined_box)
                        no_detection_count[index] = 0

                for bike in bike_boxes:
                    if intersection_over_union(person, bike) > 0.1:
                        combined_box = combine_boxes(person, bike)
                        merged_boxes.append(combined_box)
                        no_detection_count[index] = 0

            for box in merged_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw green contour
                box_id = id(box)

                if box_id not in crossed:
                    crossed[box_id] = {'line1': False, 'line2': False}
                if index == 1:
                    # Check for overlap with any part of each zone
                    if x1 < line_position_1:
                        zones_detected[0] = True
                        message_parts[0] = "Subject detected in the first part"
                    if x2 > line_position_1 and x1 < line_position_2:
                        zones_detected[1] = True
                        message_parts[1] = "Subject detected in the middle part"
                    if x2 > line_position_2:
                        zones_detected[2] = True
                        message_parts[2] = "Subject detected in the third part"
                else:
                    check_and_update_status_for_sides(frame, line_position_horizontal, y2, box_id, index)

            # Apply overlays and update messages based on zones
            for i, zone in enumerate(zones_detected):
                if zone:
                    x_start = line_position_1 * i if i > 0 else 0
                    x_end = line_position_1 * (i + 1) if i < 2 else width
                    cv2.rectangle(red_overlay, (x_start, 0), (x_end, height), (0, 0, 255), -1)
            frame = cv2.addWeighted(red_overlay, red_alpha, frame, 1 - red_alpha, 0)

            final_message = "\n".join(part for part in message_parts if part)
            if final_message:
                update_message(final_message, 1)
            else:
                no_detection_count[index] += 1
                if no_detection_count[index] >= no_detection_threshold:
                    update_message("No cyclist/biker detected", index)

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

        if caps[index].isOpened():
            ret, frame = caps[index].read()  # Read the first frame
            if index == 2:
                button_load3.grid(column=2)
            if ret:  # Check if the frame is read correctly and process the frame if needed
                frame = cv2.resize(frame, (400, 280))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert color format for Tkinter compatibility
                img = Image.fromarray(frame)  # Convert the frame to PIL format
                imgtk = ImageTk.PhotoImage(image=img)  # Convert to PhotoImage

                # Update the corresponding label to display the frame
                if index == 0:
                    label_video1.config(image=imgtk)
                    label_video1.image = imgtk  # Keep a reference to avoid garbage collection
                elif index == 1:
                    label_video2.config(image=imgtk)
                    label_video2.image = imgtk
                elif index == 2:
                    label_video3.config(image=imgtk)
                    label_video3.image = imgtk


button_load1 = ttk.Button(root, text="Load Video 1", style='TButton', command=lambda: load_video(0))
button_load1.grid(row=2, column=0, pady=20)
button_load2 = ttk.Button(root, text="Load Video 2", style='TButton', command=lambda: load_video(1))
button_load2.grid(row=2, column=1, pady=20)
button_load3 = ttk.Button(root, text="Load Video 3", style='TButton', command=lambda: load_video(2))
button_load3.grid(row=2, column=3, pady=20)

button_start = ttk.Button(root, text="Start Videos", style='TButton', command=update_frames)
button_start.grid(row=3, column=1, pady=20, padx=50)

# Centered widgets
root.grid_rowconfigure(1, weight=1)
root.grid_rowconfigure(3, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(3, weight=1)

root.mainloop()
