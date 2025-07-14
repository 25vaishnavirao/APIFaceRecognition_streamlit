# from flask import Flask, request, render_template
import os
import cv2
from uuid import uuid4
from datetime import datetime
import csv
import pickle
import sys
import traceback
from io import StringIO
import logging
import warnings
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import Process
import threading
from threading import Thread, Lock, Event
import time
import numpy as np
import keyboard

warnings.filterwarnings("ignore")

class DevServerWarningFilter(logging.Filter):
    def filter(self, record):
        return "This is a development server. Do not use it in a production deployment" not in record.getMessage()

# Paths and logging
if getattr(sys, 'frozen', False):
    base_path = os.path.dirname(sys.executable)
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

log_file = os.path.join(base_path, "Attendance_marking_system.log")
logging.basicConfig(filename=log_file, format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.DEBUG, filemode='a')
logger = logging.getLogger()
logger.addFilter(DevServerWarningFilter())
logger.setLevel(logging.DEBUG)

# Threaded Video Stream Class
class VideoStream:
    """
    Threaded video stream handler to capture frames from a webcam or IP camera.
    """
    def __init__(self, src):
        """
        Initialize the video stream and start the frame update thread.
        
        Args:
            src: Video source (0 for webcam or URL for IP camera).
        """
        self.stream = cv2.VideoCapture(src)
        self.ret, self.frame = self.stream.read()
        self.stopped = False
        self.lock = Lock()
        self.thread = Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        """
        Continuously update frames from the video stream in a separate thread.
        """
        while not self.stopped:
            ret, frame = self.stream.read()
            if not ret:
                self.stop()
                break
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        """
        Safely read the current frame from the video stream.
        
        Returns:
            A tuple of (ret, frame).
        """
        with self.lock:
            return self.ret, self.frame.copy() if self.ret else (False, None)

    def stop(self):
        """
        Stop the video stream and release the camera.
        """
        self.stopped = True
        self.thread.join()
        self.stream.release()

# Face Recognizer
class FaceRecognizer:
    """
    Class for handling face recognition using the InsightFace library.
    """
    def __init__(self, model_path, embeddings_path, log_dir="Attendance_Details"):
        """
        Initialize the face recognition system.
        
        Args:
            model_path: Path to the face recognition model.
            embeddings_path: Path to the stored face embeddings (pickle file).
            log_dir: Directory where attendance logs will be saved.
        """
        try:
            logger.info(" ")
            logger.info("Started Attendance Marking...")
            original_stdout = sys.stdout
            sys.stdout = StringIO()
            self.model = FaceAnalysis(name="buffalo_sc", root=model_path)
            self.model.prepare(ctx_id=-1)
            sys.stdout = original_stdout
            self.model_path=model_path
            self.embeddings_path = embeddings_path
            self.database_persons=self.load_database()
            self.log_dir = log_dir
            self.attendance_set = set()
            self.last_log_time = datetime.now()
            self.last_image_time = {}
            self.file_path = None
        except Exception:
            logger.exception("Error initializing FaceRecognizer.")
            traceback.print_exc()


    def get_face_embedding(self, img):
        """
        Extract facial embeddings from an image.
        
        Args:
            img: The image frame in BGR format.
        
        Returns:
            List of tuples containing embeddings and bounding boxes.
        """
        try:
            if img is None:
                return []
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = self.model.get(img_rgb)
            return [(face.embedding, face.bbox.astype(int)) for face in faces]
        except Exception as e:
            logger.error("Error getting face embedding: %s", e)
            return []
        
    def load_database(self):
        """
        Load the known face embeddings from the pickle file.
        
        Returns:
            A dictionary with names as keys and list of embeddings as values.
        """
        try:
            if not os.path.exists(self.embeddings_path):
                logger.error("Embeddings file not found.")
                return {}
            with open(self.embeddings_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.error("Error loading embeddings: %s", e)
            return {}

    def recognize_face(self, live_embedding, threshold=0.5):
        """
        Compare a live face embedding with known embeddings.
        
        Args:
            live_embedding: The embedding of the detected face.
            threshold: Minimum similarity score to consider a match.
        
        Returns:
            Tuple containing name of the recognized person and similarity score.
        """
        all_embeddings = []
        names = []
        for name, embeddings in self.database_persons.items():
            all_embeddings.extend(embeddings)
            names.extend([name] * len(embeddings))
        if not all_embeddings:
            return "Unknown", 0.0
        all_embeddings = np.array(all_embeddings)
        similarities = cosine_similarity([live_embedding], all_embeddings)[0]
        best_index = np.argmax(similarities)
        best_similarity = similarities[best_index]
        return (names[best_index], best_similarity) if best_similarity > threshold else ("Unknown", best_similarity)

    def create_new_file(self):
        """
        Create a new CSV file for logging attendance with a timestamped filename.
        """
        try:
            if self.log_dir:
                os.makedirs(self.log_dir, exist_ok=True)
            timestamp_folder = datetime.now().strftime("Attendance_%d-%m-%Y_%H-%M")
            self.session_folder = os.path.join(self.log_dir, timestamp_folder)
            os.makedirs(os.path.join(self.session_folder, "Recognized_images"), exist_ok=True)
            session_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + str(uuid4())[:8]
            self.file_path = os.path.join(self.session_folder, f"attendance_{session_id}.csv")
        except Exception as e:
            logger.error("Error creating CSV file: %s", e)

    def write(self, name, image_name, image=None):
        """
        Write attendance entry to the current session's log file.
        
        Args:
            name: The name of the recognized person.
        """
        def save_image(image, image_name):
            image_dir = os.path.join(self.session_folder, "Recognized_images")
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, image_name)
            if image is not None:
                cv2.imwrite(image_path, image)
                logger.info(f"Saved image for {name} at {image_path}")

        try:
            current_time = datetime.now()

            if self.file_path is None:
                self.create_new_file()

            if (current_time - self.last_log_time).total_seconds() >= 60:
                self.last_log_time = current_time
                self.last_image_time = {}
                self.attendance_set.clear()
                self.file_path = None
                return

            if name not in self.last_image_time or (current_time - self.last_image_time[name]).total_seconds() >= 60:
                self.last_image_time[name] = current_time
                if name not in self.attendance_set:
                    self.attendance_set.add(name)
                    timestamp = current_time.strftime('%Y-%m-%d %H:%M:%S')
                    is_new_file = not os.path.exists(self.file_path)
                    with open(self.file_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        if is_new_file:
                            writer.writerow(["Name", "DateTime", "Image Path"])
                        image_path = os.path.join(self.session_folder, "Recognized_images", image_name)
                        writer.writerow([name, timestamp, image_path])
                save_image(image, image_name)
        except Exception as e:
            logger.error("Error writing to attendance file: %s", e)

    def recognition_worker(self, ip, stop_event):
        """
        Main loop to perform live face detection and recognition from a video stream.
        
        Args:
            ip: Camera source URL or device index.
        """
        persons = []
        video_stream = VideoStream(int(ip)) if str(ip).isdigit() else VideoStream(ip)
        logger.info("Camera started: %s", ip)
        prev_time = datetime.now()

        while not stop_event.is_set():
            ret, frame = video_stream.read()
            if not ret:
                logger.warning("Failed to read frame from stream.")
                break

            current_time = datetime.now()
            elapsed = (current_time - prev_time).total_seconds()
            prev_time = current_time

            detected_faces = self.get_face_embedding(frame)
            for embedding, (x, y, w, h) in detected_faces:
                name, similarity = self.recognize_face(embedding)
                if name != "Unknown":
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (w, h), color, 2)
                    cv2.putText(frame, f"{name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    image_filename = f"{name}_{current_time.strftime('%Y%m%d_%H%M')}.jpg"
                    self.write(name, image_filename, frame)
                    persons.append(name)

            # cv2.imshow("Face Recognition", cv2.resize(frame, (1080, 720)))
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        video_stream.stop()
        # cv2.destroyAllWindows()
        logger.info("Session ended. Recognized: %s", persons)

    


    def recognize(self, ip):  
        """
        Start the face recognition process in a separate process.
        
        Args:
            ip: Camera source URL or device index.
        
        Returns:
            A list with a message indicating the recognition has started.
        """
        p = Process(target=run_recognition, args=(self.model_path, self.embeddings_path, self.log_dir, ip))
        p.start()
        logger.info("Face recognition process started.")
        return ["Face recognition process started in background."]

def run_recognition(model_path, embeddings_path, log_dir, ip):
    recognizer = FaceRecognizer(model_path, embeddings_path, log_dir)
    stop_event = Event()
    recognizer.recognition_worker(ip, stop_event)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    ip_address = sys.argv[1] if len(sys.argv) > 1 else 0
    recognizer = FaceRecognizer("./insightface_model", "embeddings.pkl", "./Attendance_Details")
    recognizer.recognize(ip_address)
