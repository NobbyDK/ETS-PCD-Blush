#!/usr/bin/env python3
"""
UDP Webcam Server - Enhanced Blush On with Stable CV2 Detection
VERSI GABUNGAN:
- Deteksi Wajah Stabil (Haar Cascade + Temporal Smoothing)
- Deteksi Landmark Presisi (LBF .yaml)
"""

import cv2
import socket
import struct
import threading
import time
import math
import numpy as np
from scipy.ndimage import gaussian_filter
from collections import deque
from pathlib import Path

class UDPWebcamServer:
    def __init__(self, host='127.0.0.1', port=8888, control_port=8889):
        self.host = host
        self.port = port
        self.control_port = control_port
        self.server_socket = None
        self.control_socket = None
        self.clients = set()
        self.camera = None
        self.running = False
        self.sequence_number = 0

        # Optimized settings
        self.max_packet_size = 32768
        self.target_fps = 15
        self.jpeg_quality = 40
        self.frame_width = 640
        self.frame_height = 480

        # Performance monitoring
        self.frame_send_time = 1.0 / self.target_fps

       # --- PERUBAHAN: Muat Cascade Classifier Kustom dari Root ---
        print("🔄 Memuat model Cascade Classifier kustom (.xml)...")
        # Langsung gunakan nama file karena ada di root
        cascade_model_path = Path("cascade_haar.xml") # <-- PATH DISESUAIKAN

        try:
            if not cascade_model_path.exists():
                print(f"❌ FATAL ERROR: File model cascade '{cascade_model_path}' tidak ditemukan.")
                print("   Pastikan file ada di folder yang sama dengan skrip ini.")
                exit()

            # Muat cascade kustom Anda
            self.face_cascade = cv2.CascadeClassifier(str(cascade_model_path))

            if self.face_cascade.empty():
                print(f"❌ FATAL ERROR: Gagal memuat cascade classifier dari {cascade_model_path}")
                exit()
            print(f"✅ Model cascade kustom '{cascade_model_path.name}' berhasil dimuat.")

            # Hapus loading HOG+SVM
            # print("🔄 Loading custom face detector model (.pkl)...")
            # self.face_detector = joblib.load("my_face_detector.pkl")
            # self.hog = cv2.HOGDescriptor()
            # print("✅ Custom face detector loaded successfully!")
            # print("✅ OpenCV face detectors initialized successfully") # Pesan ini mungkin tidak relevan lagi

            # --- Inisialisasi LBF (dari Root) ---
            print("🔄 Loading OpenCV LBF Landmark model (.yaml)...")
            lbf_model_path = Path("lbfmodel.yaml") # <-- PATH DISESUAIKAN
            if not lbf_model_path.exists():
                print(f"❌ FATAL ERROR: File landmark '{lbf_model_path}' tidak ditemukan.")
                print("   Pastikan file ada di folder yang sama dengan skrip ini.")
                exit()
            self.landmark_detector = cv2.face.createFacemarkLBF()
            self.landmark_detector.loadModel(str(lbf_model_path))
            print(f"✅ LBF model '{lbf_model_path.name}' loaded successfully.")

        except cv2.error as e:
            print(f"❌ FATAL ERROR: Gagal memuat model OpenCV.")
            print(f"   Pastikan file '{cascade_model_path.name}' dan '{lbf_model_path.name}' ada dan valid.")
            print(f"   OpenCV Error: {e}")
            exit()
        except Exception as e:
            print(f"❌ Error loading OpenCV models: {e}")
            raise e

        # --- Temporal Smoothing  ---
        self.face_history = deque(maxlen=5)  # Store last 5 face detections
        # (Cheek history dihapus, kita pakai landmark langsung)
        self.last_valid_face = None
        self.frames_without_detection = 0
        self.max_frames_without_detection = 10
        
        # --- Blush Settings ---
        self.blush_color_rgb = (235, 148, 146)
        self.blush_intensity = 0.25
        self.blush_blur = 15
        self.lock = threading.Lock()

    def initialize_camera(self):
        print("🎥 Initializing optimized camera...")
        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if self.camera.isOpened():
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.camera.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            print(f"✅ Camera initialized: {actual_width}x{actual_height} @ {actual_fps:.2f} FPS")
            return True
        else:
            print("❌ Failed to initialize camera")
            return False

    def smooth_face_detection(self, current_face):
        """
        Smooth face detection using temporal averaging
        """
        if current_face is None:
            self.frames_without_detection += 1
            
            if self.frames_without_detection < self.max_frames_without_detection and self.last_valid_face is not None:
                return self.last_valid_face
            else:
                return None
        
        self.frames_without_detection = 0
        self.face_history.append(current_face)
        
        if len(self.face_history) > 0:
            avg_x = int(np.mean([f[0] for f in self.face_history]))
            avg_y = int(np.mean([f[1] for f in self.face_history]))
            avg_w = int(np.mean([f[2] for f in self.face_history]))
            avg_h = int(np.mean([f[3] for f in self.face_history]))
            
            smoothed_face = (avg_x, avg_y, avg_w, avg_h)
            self.last_valid_face = smoothed_face
            return smoothed_face
        
        return current_face

    def detect_face_robust(self, gray_frame):
        """
        Deteksi wajah menggunakan Cascade Classifier kustom (.xml)
        """
        # (Opsional) Tingkatkan kontras
        gray_eq = cv2.equalizeHist(gray_frame)

        # Parameter deteksi (sesuaikan jika perlu)
        scaleFactor = 1.05   # Coba 1.05 hingga 1.3
        minNeighbors = 4    # Coba 3 hingga 6
        minSize = (32, 32)  # Sesuaikan dengan ukuran training Anda

        # Gunakan self.face_cascade yang sudah dimuat di __init__
        # PASTIKAN TIDAK ADA LAGI REFERENSI KE self.face_cascade_alt di sini
        faces = self.face_cascade.detectMultiScale(
            gray_eq, # Gunakan gambar yg sudah di-equalize
            scaleFactor=scaleFactor,
            minNeighbors=minNeighbors,
            minSize=minSize
        )

        # Kembalikan array numpy jika ada deteksi, None jika tidak
        return faces if len(faces) > 0 else None
    
    # def detect_face_robust(self, gray_frame):
    #     h, w = gray_frame.shape
    #     win_size = (64, 128)
    #     step = 8
    #     faces = []

    #     for y in range(0, h - win_size[1], step):
    #         for x in range(0, w - win_size[0], step):
    #             window = gray_frame[y:y + win_size[1], x:x + win_size[0]]
    #             hog_desc = self.hog.compute(window).flatten()

    #             pred = self.face_detector.predict([hog_desc])

    #             if pred == 1:  # Wajah
    #                 faces.append((x, y, win_size[0], win_size[1]))

    #     if len(faces) > 0:
    #         # Pilih wajah terbesar (umumnya wajah utama)
    #         faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    #         return faces

    #     return None


    # --- (.yaml) ---
    def get_cheek_contour_points(self, face_landmarks_cv2, is_left=True):
        """
        Mendapatkan titik-titik kontur pipi menggunakan 68 landmarks LBF (CV2).
        """
        if is_left:
            # Pipi Kiri (indeks 68)
            indices = [
            1, 2, 3,        # Area tulang pipi atas kiri
            31,         # Area dekat hidung dan mulut
            39,             # Area tengah pipi
            28              # Area dekat hidung (untuk batas atas)
        ]
        else:
            # Pipi Kanan (indeks 68)
            indices = [
            15, 14, 13,     # Area tulang pipi atas kanan
            35,         # Area dekat hidung dan mulut
            42,             # Area tengah pipi
            28              # Area dekat hidung (untuk batas atas)
        ]

        points = []
        for idx in indices:
            point = face_landmarks_cv2[idx]
            points.append((int(point[0]), int(point[1]))) # (x, y)

        return np.array(points, dtype=np.int32)

    # --- (.yaml) ---
    def create_smooth_blush_mask(self, frame_shape, points, blur_radius):
        """
        Membuat mask blush yang smooth dengan convex hull (cocok untuk landmarks)
        MENGGUNAKAN CV2.BLUR (LEBIH CEPAT)
        """
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        
        if len(points) < 3:
            return mask
        
        try:
            # Buat convex hull dari points untuk area blush
            hull = cv2.convexHull(points)
            cv2.fillConvexPoly(mask, hull, 1.0)
        except cv2.error:
             return mask # Gagal jika poin kolinier

        # --- PERGANTIAN: Gunakan cv2.blur (jauh lebih cepat) ---
        if blur_radius > 0:
            # Ukuran kernel harus ganjil untuk hasil yang simetris
            ksize = blur_radius * 2 + 1 
            mask = cv2.blur(mask, (ksize, ksize))
        # --------------------------------------------------

        # Normalize mask
        max_val = mask.max()
        if max_val > 0:
            mask = mask / max_val
        
        return mask

    def apply_blush(self, frame):
        """
        FUNGSI GABUNGAN - MULTI FACE VERSION:
        1. Deteksi SEMUA wajah stabil (Haar)
        2. Deteksi landmark presisi (LBF) untuk setiap wajah
        3. Aplikasi blush pada semua wajah
        """
        output_frame = frame.copy().astype(np.float32)
        
        # 1. Konversi ke grayscale 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Deteksi SEMUA wajah 
        all_faces = self.detect_face_robust(gray)
        
        if all_faces is not None and len(all_faces) > 0:
            h, w, _ = frame.shape
            
            # Get current settings
            with self.lock:
                color_rgb = self.blush_color_rgb
                intensity = self.blush_intensity
                blur = self.blush_blur
            
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
            
            # LBF .fit() needs face list as numpy array
            faces_array = np.array(all_faces)
            
            try:
                ok, landmarks_list = self.landmark_detector.fit(gray, faces_array)
            except cv2.error as e:
                # Gagal fit, kembalikan frame asli
                return frame
            
            if ok and landmarks_list is not None:
                # Proses SETIAP wajah yang terdeteksi
                for face_landmarks_cv2 in landmarks_list:
                    current_face_points = face_landmarks_cv2[0]

                    # 5. Dapatkan titik pipi untuk wajah ini
                    left_cheek_points = self.get_cheek_contour_points(current_face_points, is_left=True)
                    right_cheek_points = self.get_cheek_contour_points(current_face_points, is_left=False)
                    
                    # 6. Buat mask untuk wajah ini
                    left_mask = self.create_smooth_blush_mask(
                        (h, w), left_cheek_points, blur
                    )
                    right_mask = self.create_smooth_blush_mask(
                        (h, w), right_cheek_points, blur
                    )
                    
                    # 7. Blending untuk wajah ini
                    combined_mask = np.maximum(left_mask, right_mask)
                    combined_mask = combined_mask * intensity
                    
                    color_overlay = np.zeros_like(output_frame)
                    color_overlay[:, :] = color_bgr
                    
                    for c in range(3):
                        output_frame[:, :, c] = (
                            output_frame[:, :, c] * (1 - combined_mask) +
                            color_overlay[:, :, c] * combined_mask
                        )
            
        return output_frame.astype(np.uint8)

    def listen_for_clients(self):
        """ Listens for incoming client messages """
        print(f"👂 Listening for clients on {self.host}:{self.port}...")
        while self.running:
            try:
                data, addr = self.server_socket.recvfrom(1024)
                message = data.decode('utf-8')

                if addr not in self.clients and message == "REGISTER":
                    print(f"➕ New client connected: {addr}")
                    self.clients.add(addr)
                    self.server_socket.sendto(b"REGISTERED", addr)
                    print(f"✅ Sent registration confirmation to {addr}")
                elif message == "UNREGISTER" and addr in self.clients:
                    print(f"➖ Client disconnected: {addr}")
                    self.clients.discard(addr)

            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"❌ Listener error: {e}")
                break

    def listen_for_controls(self):
        """ Listens for control commands to update blush settings """
        print(f"🎮 Control socket listening on {self.host}:{self.control_port}...")
        while self.running:
            try:
                data, addr = self.control_socket.recvfrom(1024)
                command = data.decode('utf-8').strip()
                
                if command.startswith("COLOR:"):
                    try:
                        rgb_str = command.split(":")[1]
                        r, g, b = map(int, rgb_str.split(","))
                        with self.lock:
                            self.blush_color_rgb = (r, g, b)
                        print(f"🎨 Blush color updated to RGB({r}, {g}, {b})")
                        self.control_socket.sendto(b"COLOR_OK", addr)
                    except Exception as e:
                        print(f"❌ Invalid color format: {e}")
                        self.control_socket.sendto(b"COLOR_ERROR", addr)
                
                elif command.startswith("INTENSITY:"):
                    try:
                        intensity = float(command.split(":")[1])
                        intensity = max(0.0, min(1.0, intensity))
                        with self.lock:
                            self.blush_intensity = intensity
                        print(f"💪 Blush intensity updated to {intensity:.2f}")
                        self.control_socket.sendto(b"INTENSITY_OK", addr)
                    except Exception as e:
                        print(f"❌ Invalid intensity format: {e}")
                        self.control_socket.sendto(b"INTENSITY_ERROR", addr)
                
                elif command.startswith("BLUR:"):
                    try:
                        blur = int(command.split(":")[1])
                        blur = max(5, min(50, blur))
                        with self.lock:
                            self.blush_blur = blur
                        print(f"🌫️ Blush blur updated to {blur}px")
                        self.control_socket.sendto(b"BLUR_OK", addr)
                    except Exception as e:
                        print(f"❌ Invalid blur format: {e}")
                        self.control_socket.sendto(b"BLUR_ERROR", addr)

            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"❌ Control listener error: {e}")

    def send_frames(self):
        """ Captures frames, applies blush, encodes, packets, and sends them """
        print("🚀 Starting frame broadcast...")

        while self.running:
            start_capture_time = time.time()

            ret, frame = self.camera.read()
            if not ret:
                print("⚠️ Dropped frame")
                time.sleep(0.01)
                continue

            # Apply Blush (menggunakan fungsi gabungan)
            frame_with_blush = self.apply_blush(frame)

            # Encode to JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            result, encoded_frame = cv2.imencode('.jpg', frame_with_blush, encode_param)

            if not result:
                print("❌ JPEG encoding failed")
                continue

            frame_data = encoded_frame.tobytes()
            frame_size = len(frame_data)
            self.sequence_number += 1

            # Packetization
            header_size = 12
            payload_size = self.max_packet_size - header_size
            total_packets = math.ceil(frame_size / payload_size)

            # Send to all clients
            current_clients = self.clients.copy()
            for client_addr in current_clients:
                try:
                    for packet_index in range(total_packets):
                        start_pos = packet_index * payload_size
                        end_pos = min(start_pos + payload_size, frame_size)

                        header = struct.pack("!III", self.sequence_number, total_packets, packet_index)
                        udp_packet = header + frame_data[start_pos:end_pos]

                        self.server_socket.sendto(udp_packet, client_addr)

                    if self.sequence_number % (self.target_fps * 2) == 1:
                        print(f"📤 Frame {self.sequence_number} ({frame_size // 1024} KB) → {len(current_clients)} clients")

                except socket.error as se:
                    print(f"❌ Socket error sending to {client_addr}: {se}")
                    self.clients.discard(client_addr)
                except Exception as e:
                    print(f"❌ Unexpected error sending to {client_addr}: {e}")
                    self.clients.discard(client_addr)

            # Frame rate control
            elapsed_time = time.time() - start_capture_time
            sleep_time = self.frame_send_time - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    def start_server(self):
        if self.running:
            print("⚠️ Server already running")
            return

        print(f"🟢 Starting UDP server on {self.host}:{self.port}...")
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.settimeout(0.5)

        print(f"🎮 Starting control socket on {self.host}:{self.control_port}...")
        self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.control_socket.bind((self.host, self.control_port))
        self.control_socket.settimeout(0.5)

        if not self.initialize_camera():
            self.server_socket.close()
            self.control_socket.close()
            return

        self.running = True
        self.sequence_number = 0

        self.listener_thread = threading.Thread(target=self.listen_for_clients, daemon=True)
        self.listener_thread.start()

        self.control_thread = threading.Thread(target=self.listen_for_controls, daemon=True)
        self.control_thread.start()

        try:
            self.send_frames()
        except KeyboardInterrupt:
            print("\n⌨️ Ctrl+C detected. Stopping server...")
        finally:
            self.stop_server()

    def stop_server(self):
        print("⏹️ Stopping server...")
        self.running = False
        time.sleep(0.1)

        if self.server_socket:
            self.server_socket.close()
            self.server_socket = None
        if self.control_socket:
            self.control_socket.close()
            self.control_socket = None
        if self.camera:
            self.camera.release()
            self.camera = None

        print("✅ Server stopped")

if __name__ == "__main__":
    print("=== Hybrid CV2 Blush Server (Stable Haar + Precise LBF) ===")
    print("📝 Dependencies: pip install opencv-python-contrib scipy numpy")
    print("⚠️  Pastikan 'lbfmodel.yaml' ada di folder!")
    print("✨ Features:")
    print("    - Temporal smoothing for stable tracking")
    print("    - Multi-cascade face detection")
    print("    - LBF Landmark-based cheek positioning")
    
    server = UDPWebcamServer()
    try:
        server.start_server()
    except Exception as e:
        print(f"💥 Unhandled exception: {e}")
        server.stop_server()