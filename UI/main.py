from tkinter import Tk, Button, filedialog
import cv2
from ultralytics import YOLO  # Ensure correct import for your YOLO model usage

# Initialize the YOLO model
model = YOLO("yolov8n.pt")


# Function to load and process video
def load_video():
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    if not video_path:
        return

    cap = cv2.VideoCapture(video_path)

    ret = True
    while ret:
        ret, frame = cap.read()
        if ret:
            # Resize frame to make it smaller
            frame = cv2.resize(frame, (640, 360))  # Resize to width=640, height=360

            # Cut the bottom 25% of the video frame
            height = frame.shape[0]
            frame = frame[:int(height * 0.80)]  # Keep only the top 75% of the frame
            height = frame.shape[0]  # Update the height after cropping

            # Add semi-transparent horizontal lines lower on the frame
            line_color = (0, 0, 0)  # Green color for the lines
            line_thickness = 2
            alpha = 0.3  # Transparency factor
            overlay = frame.copy()
            line_positions = [int(height * 0.9), int(height * 0.8),
                              int(height * 0.7)]  # Positions at 60%, 70%, and 80% height

            for pos in line_positions:
                cv2.line(overlay, (0, pos), (frame.shape[1], pos), line_color, line_thickness)

            # Blend the overlay with the original frame
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # Detect objects of specific classes (0 - person, 1 - bicycle, 3 - motorcycle)
            results = model.predict(frame, classes=[0, 1, 3])

            if results:
                frame_ = results[
                    0].plot()  # Assuming 'results' can be handled like this; adjust according to your actual object

                # Visualize
                cv2.imshow('Video Display', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()


# Initialize the GUI
root = Tk()
root.title("Video Loader")
root.geometry('800x600')  # Set the size of the window to 800x600 pixels

# Create a button to load video
btn_load = Button(root, text="Load Video", command=load_video)
btn_load.pack(pady=20)

root.mainloop()
