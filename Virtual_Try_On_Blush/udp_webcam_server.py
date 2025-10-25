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

        # --- Enhanced CV2 Face Detection (DARI KODE BARU ANDA) ---
        print("🔄 Loading OpenCV Haar Cascade classifiers...")
        
        try:
            # Load multiple cascades for better detection
            # PENTING: Kita menggunakan cv2.data.haarcascades
            # Ini berarti Anda TIDAK PERLU mengunduh file .xml ini
            # Asumsi OpenCV terinstal dengan benar
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.face_cascade_alt = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
            )
            # --- (haarcascade_eye.xml DIHAPUS, diganti LBF) ---
            
            print("✅ OpenCV face detectors initialized successfully")
            
            # --- BARU: Inisialisasi LBF (DARI KODE LAMA) ---
            print("🔄 Loading OpenCV LBF Landmark model (.yaml)...")
            # PASTIKAN FILE 54MB INI ADA DI FOLDER YANG SAMA!
            model_path = "lbfmodel.yaml" 
            self.landmark_detector = cv2.face.createFacemarkLBF()
            self.landmark_detector.loadModel(model_path)
            print(f"✅ LBF model '{model_path}' loaded successfully.")

        except cv2.error as e:
            print(f"❌ FATAL ERROR: Gagal memuat model 'lbfmodel.yaml'.")
            print("Pastikan Anda menggunakan file 68-point yang berukuran ~54MB.")
            print(f"OpenCV Error: {e}")
            exit()
        except Exception as e:
            print(f"❌ Error loading OpenCV models: {e}")
            raise e

        # --- Temporal Smoothing (DARI KODE BARU ANDA) ---
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
        # (Fungsi ini sama persis dengan kode baru Anda)
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
        # (Fungsi ini sama persis dengan kode baru Anda)
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
        # (Fungsi ini sama persis dengan kode baru Anda)
        """
        Robust face detection with multiple methods and frame preprocessing
        """
        gray_eq = cv2.equalizeHist(gray_frame)
        gray_blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)
        
        faces = self.face_cascade.detectMultiScale(
            gray_blur,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) == 0:
            faces = self.face_cascade_alt.detectMultiScale(
                gray_blur,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(60, 60)
            )
        
        if len(faces) > 0:
            faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            return faces_sorted[0] # Return (x, y, w, h)
        
        return None

    # --- BARU: Ditambahkan dari kode LAMA (.yaml) ---
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

    # --- BARU: Ditambahkan dari kode LAMA (.yaml) ---
    def create_smooth_blush_mask(self, frame_shape, points, blur_radius):
        """
        Membuat mask blush yang smooth dengan convex hull (cocok untuk landmarks)
        """
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        
        if len(points) < 3:
            return mask
        
        # Buat convex hull dari points untuk area blush
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 1.0)
        
        # Apply gaussian blur untuk smooth transition
        mask = gaussian_filter(mask, sigma=blur_radius)
        
        # Normalize mask
        if mask.max() > 0:
            mask = mask / mask.max()
        
        return mask

    def apply_blush(self, frame):
        """
        FUNGSI GABUNGAN:
        1. Deteksi wajah stabil (Haar)
        2. Deteksi landmark presisi (LBF)
        3. Aplikasi blush
        """
        output_frame = frame.copy().astype(np.float32)
        
        # 1. Konversi ke grayscale (DARI KODE BARU)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Deteksi wajah (DARI KODE BARU)
        current_face = self.detect_face_robust(gray)
        
        # 3. Smoothing wajah (DARI KODE BARU)
        face_rect = self.smooth_face_detection(current_face)
        
        if face_rect is not None:
            h, w, _ = frame.shape
            
            # --- 4. LOGIKA LBF (DARI KODE LAMA) DIMASUKKAN KE SINI ---
            # LBF .fit() butuh daftar wajah, jadi kita bungkus
            faces_list = np.array([face_rect]) 
            
            try:
                ok, landmarks_list = self.landmark_detector.fit(gray, faces_list)
            except cv2.error as e:
                # Gagal fit, kembalikan frame asli
                return frame
            
            if ok and landmarks_list is not None:
                # Get current settings
                with self.lock:
                    color_rgb = self.blush_color_rgb
                    intensity = self.blush_intensity
                    blur = self.blush_blur
                
                color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

                # Proses landmark untuk wajah yang terdeteksi
                for face_landmarks_cv2 in landmarks_list:
                    current_face_points = face_landmarks_cv2[0]

                    # 5. Dapatkan titik pipi (DARI KODE LAMA)
                    left_cheek_points = self.get_cheek_contour_points(current_face_points, is_left=True)
                    right_cheek_points = self.get_cheek_contour_points(current_face_points, is_left=False)
                    
                    # 6. Buat mask (DARI KODE LAMA)
                    left_mask = self.create_smooth_blush_mask(
                        (h, w), left_cheek_points, blur
                    )
                    right_mask = self.create_smooth_blush_mask(
                        (h, w), right_cheek_points, blur
                    )
                    
                    # 7. Blending (LOGIKA UMUM)
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

    # --- SISA FUNGSI (NETWORKING) SAMA PERSIS ---

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
    # PENTING: Anda perlu opencv-python-contrib untuk cv2.face
    print("📝 Dependencies: pip install opencv-python-contrib scipy numpy")
    print("⚠️  Pastikan 'lbfmodel.yaml' (versi 54MB) ada di folder!")
    print("✨ Features:")
    print("    - Temporal smoothing for stable tracking (dari kode baru)")
    print("    - Multi-cascade face detection (dari kode baru)")
    print("    - LBF Landmark-based cheek positioning (dari kode lama)")
    
    server = UDPWebcamServer()
    try:
        server.start_server()
    except Exception as e:
        print(f"💥 Unhandled exception: {e}")
        server.stop_server()