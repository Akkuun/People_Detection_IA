import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import Button, Label, Frame, filedialog, StringVar
from PIL import Image, ImageTk
import os
import time
from ultralytics import YOLO
from Utility.CaptureFace import CaptureFace
from Utility.MugshotPipeline import MugshotPipeline

color_grey = "#f4f4f4"
color_preview = "#e9ecef"
color_button_bg = "#ffffff"
color_button_fg = "#333"
color_button_active = "#e0e0e0"
color_info_fg = "#555"
color_folder_fg = "#0078d7"

class WebcamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Webcam Detection - Mugshot")
        self.root.configure(bg=color_grey)
        self.root.geometry("1200x800") # Taille actuel TODO à ajuster selon le besoin
        self.root.resizable(False, False)

        # Images sauvegardées dans le dossier SavedImages par défaut (du projet)
        self.save_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "../SavedImages"))
        self.save_folder_var = StringVar(value=self.save_folder)

        # On centre le contenu
        container = Frame(root, bg=color_grey)
        container.pack(expand=True)

        # Preview de la webcam
        self.image_label = Label(container, bg=color_preview, bd=2, relief="groove")
        self.image_label.pack(pady=40)

        # Sélection du dossier de sauvegarde (bouton et label)
        folder_frame = Frame(container, bg=color_grey)
        folder_frame.pack(pady=5)
        Label(folder_frame, text="Save folder:", font=("Segoe UI", 12), bg=color_grey, fg=color_info_fg).pack(side=tk.LEFT)
        self.folder_path_label = Label(folder_frame, textvariable=self.save_folder_var, font=("Segoe UI", 12), bg=color_grey, fg=color_folder_fg)
        self.folder_path_label.pack(side=tk.LEFT, padx=10)
        select_btn = Button(folder_frame, text="Change...", command=self.select_folder, font=("Segoe UI", 11), bg=color_button_bg, fg=color_button_fg, bd=0, padx=10, pady=5, relief="ridge", activebackground=color_button_active)
        select_btn.pack(side=tk.LEFT, padx=10)

        # ScreenShot bouton
        btn_frame = Frame(container, bg=color_grey)
        btn_frame.pack(pady=10)
        self.screenshot_btn = Button(
            btn_frame, text="Screenshot", command=self.save_screenshot,
            font=("Segoe UI", 14), bg=color_button_bg, fg=color_button_fg, bd=0, padx=20, pady=10, relief="ridge", activebackground=color_button_active
        )
        self.screenshot_btn.pack(side=tk.LEFT, padx=20)
        # Mugshot bouton
        self.mugshot_btn = Button(
            btn_frame, text="Mugshot", command=self.generate_mugshot,
            font=("Segoe UI", 14), bg=color_button_bg, fg=color_button_fg, bd=0, padx=20, pady=10, relief="ridge", activebackground=color_button_active
        )
        self.mugshot_btn.pack(side=tk.LEFT, padx=20)

        # label notifications
        self.info_label = Label(container, text="Click Screenshot to capture, Mugshot to generate.",
                               font=("Segoe UI", 12), bg=color_grey, fg=color_info_fg)
        self.info_label.pack(pady=10)

        self.cap = cv2.VideoCapture(0)
        self.frame = None
        self.annotated = None
        self.results = None
        self.model = YOLO("yolov8n.pt", verbose=False)
        self.mugshot_generator = CaptureFace()
        self.mugshot_pipeline = MugshotPipeline()
        self.update_frame()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind('<Escape>', lambda event: self.on_close()) # Fermer l'application avec la touche ESC

    # Change les variables du dossier de sauvegarde
    def select_folder(self):
        folder = filedialog.askdirectory(initialdir=self.save_folder)
        if folder:
            self.save_folder = folder
            self.save_folder_var.set(folder)

    # Met à jour le flux vidéo via la webcam et effectue la détection YOLO via draw_boxes_opencv
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame = frame
            try:
                self.results = self.model(frame, verbose=False)
            except Exception as e:
                self.annotated = frame.copy()
            else:
                self.annotated = frame.copy()
                self.draw_boxes_opencv(self.annotated, self.results)
            rgb_image = cv2.cvtColor(self.annotated, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.image_label.imgtk = imgtk
            self.image_label.configure(image=imgtk)
        self.root.after(30, self.update_frame)

    def draw_boxes_opencv(self, frame, results):
        if results is None or len(results) == 0:
            return
        res = results[0]
        boxes = getattr(res, "boxes", None)
        if boxes is None:
            return
        try:
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
            confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
            clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.array(boxes.cls)
        except Exception:
            return
        for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clss):
            try:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            except Exception:
                continue
            label_name = self.model.names[int(cls)] if hasattr(self.model, "names") and self.model.names is not None else str(int(cls))
            label = f"{label_name} {float(conf):.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (16, 255, 16), 2)
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), (16, 255, 16), -1)
            cv2.putText(frame, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    def save_screenshot(self):
        if self.annotated is not None:
            os.makedirs(self.save_folder, exist_ok=True)
            path = os.path.join(self.save_folder, "screenshot.jpg")
            cv2.imwrite(path, self.annotated)
            self.info_label.config(text=f"Screenshot saved: {path}")

    # Fais appel à la pipeline de génération de mugshot pour chaque personne détectée (MugshotPipeline)
    def generate_mugshot(self):
        if self.results is None or len(self.results) == 0:
            self.info_label.config(text="No detection results available.")
            return
        res = self.results[0]
        boxes = getattr(res, "boxes", None)
        if boxes is not None:
            try:
                xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
                clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.array(boxes.cls)
                mugshot_count = 0
                timestamp = int(time.time())
                for i, (box, cls) in enumerate(zip(xyxy, clss)):
                    if int(cls) == 0:  # Person detected (class 0 in YOLO)
                        x1, y1, x2, y2 = map(int, box)
                        bbox = (x1, y1, x2-x1, y2-y1)
                        # Récupération du visage via CaptureFace
                        face = self.mugshot_generator.extract_face_from_detection(self.frame, bbox)
                        if face is not None:
                            # Récupération de l'orientation et traitement du mugshot via MugshotPipeline
                            processed_face, orientation = self.mugshot_pipeline.process_mugshot(face)
                            if processed_face is not None:
                                os.makedirs(self.save_folder, exist_ok=True)
                                filename = f'mugshot_{orientation}_{timestamp}_person{i}.jpg'
                                path = os.path.join(self.save_folder, filename)
                                cv2.imwrite(path, processed_face)
                                mugshot_count += 1
                if mugshot_count > 0:
                    self.info_label.config(text=f"{mugshot_count} mugshot(s) generated!")
                else:
                    self.info_label.config(text="No person detected for mugshot generation.")
            except Exception as e:
                self.info_label.config(text=f"Error generating mugshot.")
        else:
            self.info_label.config(text="No detection available.")

    def on_close(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = WebcamApp(root)
    root.mainloop()