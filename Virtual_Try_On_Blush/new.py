#!/usr/bin/env python3
"""
UDP Webcam Server - Enhanced Blush On with Stable CV2 Detection
VERSI GABUNGAN:
- Deteksi Wajah Stabil (HOG+SVM Kustom)
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
import joblib
from skimage.feature import hog  # <-- DITAMBAHKAN

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

        # --- Enhanced CV2 Face Detection ---
        print("🔄 Loading OpenCV Haar Cascade classifiers...")
        
        try:
            # # Load multiple cascades for better detection (DINONAKTIFKAN, KITA PAKAI KUSTOM)
            # self.face_cascade = cv2.CascadeClassifier(
            #     cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            # )
            # self.face_cascade_alt = cv2.CascadeClassifier(
            #     cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
            # )
            
            # --- PERBAIKAN 1: Muat model HOG+SVM kustom ---
            print("🔄 Loading custom face detector model (.pkl)...")
            model_path = "face_detector_hog_svm.pkl" # Pastikan nama file ini benar
            
            try:
                model_data = joblib.load(model_path)
                self.face_detector = model_data['model']  # Ambil model SVM
                self.scaler = model_data['scaler']        # Ambil StandardScaler
                self.img_size = tuple(model_data['img_size']) # Ambil ukuran (misal: (64, 64))
                print(f"✅ Custom detector (HOG+SVM) loaded from {model_path}")
                print(f"   -> Model image size: {self.img_size}")
            except FileNotFoundError:
                print(f"❌ FATAL ERROR: Model file '{model_path}' not found.")
                print("   Pastikan Anda sudah menjalankan Script 1 (training) dan nama filenya benar.")
                exit()
            except KeyError:
                print(f"❌ FATAL ERROR: File model '{model_path}' tidak valid.")
                print("   Pastikan file .pkl berisi keys: 'model', 'scaler', 'img_size'.")
                exit()
            except Exception as e:
                print(f"❌ FATAL ERROR: Gagal memuat {model_path}. Error: {e}")
                exit()

            # HAPUS: self.hog = cv2.HOGDescriptor() # Ini tidak dipakai
            
            print("✅ OpenCV face detectors initialized successfully")
            
            # --- BARU: Inisialisasi LBF ---
            print("🔄 Loading OpenCV LBF Landmark model (.yaml)...")
            model_path_lbf = "lbfmodel.yaml" 
            self.landmark_detector = cv2.face.createFacemarkLBF()
            self.landmark_detector.loadModel(model_path_lbf)
            print(f"✅ LBF model '{model_path_lbf}' loaded successfully.")

        except cv2.error as e:
            print(f"❌ FATAL ERROR: Gagal memuat model 'lbfmodel.yaml'.")
            print("Pastikan Anda menggunakan file 68-point yang berukuran ~54MB.")
            print(f"OpenCV Error: {e}")
            exit()
        except Exception as e:
            print(f"❌ Error loading OpenCV models: {e}")
            raise e

        # --- Temporal Smoothing (Tidak dipakai di detektor HOG, tapi tidak masalah) ---
        self.face_history = deque(maxlen=5)
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
        (Fungsi ini tidak lagi digunakan oleh 'detect_face_robust' versi HOG)
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

    # --- PERBAIKAN 2: Salin fungsi helper dari Script 1 ---
    
    def extract_hog_features(self, image):
        """
        Ekstraksi fitur HOG dari gambar (IDENTIK DENGAN SCRIPT 1)
        """
        # Resize image ke ukuran yang digunakan saat training
        resized = cv2.resize(image, self.img_size)
        
        # Extract HOG features (Parameter HARUS SAMA dengan training)
        features = hog(
            resized,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            visualize=False,
            feature_vector=True
        )
        return features

    def _non_max_suppression(self, detections, overlap_thresh=0.5):
        """Improved non-maximum suppression (IDENTIK DENGAN SCRIPT 1)"""
        if len(detections) == 0:
            return []
        
        boxes = np.array([d['box'] for d in detections])
        confidences = np.array([d['confidence'] for d in detections])
        
        if len(boxes) == 1:
            return detections
        
        indices = np.argsort(confidences)[::-1]
        
        keep = []
        while len(indices) > 0:
            current = indices[0]
            keep.append(current)
            
            current_box = boxes[current]
            other_boxes = boxes[indices[1:]]
            
            ious = []
            for other_box in other_boxes:
                iou = self._iou(current_box, other_box)
                ious.append(iou)
            
            ious = np.array(ious)
            
            indices = indices[1:][ious < overlap_thresh]
        
        return [detections[i] for i in keep]

    def _iou(self, box1, box2):
        """Calculate Intersection over Union (IDENTIK DENGAN SCRIPT 1)"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0

    # --- PERBAIKAN 3: Ganti total fungsi deteksi ---

    def detect_face_robust(self, gray_frame):
        """
        Deteksi wajah menggunakan model HOG+SVM kustom (dari Script 1)
        (Ini adalah salinan dari detect_faces_optimized dari Script 1)
        """
        detections = []
        # Ambil ukuran window HOG dari model yang di-load
        win_w, win_h = self.img_size 
        
        # Anda bisa sesuaikan parameter ini
        min_face_size = 100
        max_face_size = 400
        
        # Pastikan frame tidak kosong
        if min(gray_frame.shape) == 0:
            return None
            
        # --- PERBAIKAN LOGIKA SKALA ---
        # Skala terkecil (untuk deteksi wajah terbesar)
        min_scale = max(0.1, min_face_size / max(gray_frame.shape))
        
        # Skala terbesar (untuk deteksi wajah terkecil)
        max_scale = min(1.0, max_face_size / min(gray_frame.shape)) 

        
        # --- PERBAIKAN BUG ('list' <= 'float') ---
        # Cek jika skala tidak valid (misal gambar terlalu kecil)
        if max_scale <= min_scale: 
             scales = np.array([0.5]) # Buat numpy array, BUKAN list
        else:
             scales = np.linspace(min_scale, max_scale, 8)
             # Lakukan filter HANYA di dalam else block ini
             scales = scales[scales <= 1.0] 
        # -------------------------------------------
        
        for scale in scales:
            # Pastikan skala tidak nol untuk menghindari div by zero
            if scale <= 0:
                continue

            new_h = int(gray_frame.shape[0] * scale)
            new_w = int(gray_frame.shape[1] * scale)
            
            if new_h < win_h or new_w < win_w:
                continue
                
            resized = cv2.resize(gray_frame, (new_w, new_h))
            
            # Adaptive step size berdasarkan scale
            step_size = max(8, int(20 * scale))
            
            for y in range(0, resized.shape[0] - win_h, step_size):
                for x in range(0, resized.shape[1] - win_w, step_size):
                    window = resized[y:y+win_h, x:x+win_w]
                    
                    # 1. Ekstraksi fitur (HARUS SAMA DENGAN TRAINING)
                    feat = self.extract_hog_features(window)
                    # 2. Scaling (WAJIB)
                    feat_scaled = self.scaler.transform([feat])
                    # 3. Prediksi
                    confidence = self.face_detector.decision_function(feat_scaled)[0]
                    
                    if confidence > 1.5:  # (Atau nilai confidence Anda)
                            orig_x = int(x / scale)
                            orig_y = int(y / scale)
                            orig_w = int(win_w / scale)
                            orig_h = int(win_h / scale)
                            
                            # --- PERBAIKAN: Perbesar kotak secara manual ---
                            # Kita tahu wajah lebih tinggi, jadi tambah padding H lebih banyak
                            padding_w = int(orig_w * 0.20) # Tambah 20% lebar
                            padding_h = int(orig_h * 0.40) # Tambah 40% tinggi
                            
                            final_x = orig_x - (padding_w // 2)
                            final_y = orig_y - (padding_h // 2) # Geser ke atas
                            final_w = orig_w + padding_w
                            final_h = orig_h + padding_h
                            # ----------------------------------------------
                            
                            detections.append({
                                'box': [final_x, final_y, final_w, final_h], # Gunakan nilai baru
                                'confidence': confidence
                            })
        
        # Non-Maximum Suppression
        final_detections = self._non_max_suppression(detections, overlap_thresh=0.3)
        
        # Konversi ke format yang diharapkan LBF: np.array([[x,y,w,h], ...])
        faces_array = [d['box'] for d in final_detections]
        
        if len(faces_array) > 0:
            return np.array(faces_array)
        else:
            return None # Kembalikan None jika tidak ada deteksi

    # --- (Bagian Landmark dan Blush TIDAK BERUBAH) ---
    def get_cheek_contour_points(self, face_landmarks_cv2, is_left=True):
        """
        Mendapatkan titik-titik kontur pipi menggunakan 68 landmarks LBF (CV2).
        """
        if is_left:
            # Pipi Kiri (indeks 68)
            indices = [
                1, 2, 3,      # Area tulang pipi atas kiri
                31,           # Area dekat hidung dan mulut
                39,           # Area tengah pipi
                28            # Area dekat hidung (untuk batas atas)
            ]
        else:
            # Pipi Kanan (indeks 68)
            indices = [
                15, 14, 13,   # Area tulang pipi atas kanan
                35,           # Area dekat hidung dan mulut
                42,           # Area tengah pipi
                28            # Area dekat hidung (untuk batas atas)
            ]

        points = []
        for idx in indices:
            point = face_landmarks_cv2[idx]
            points.append((int(point[0]), int(point[1]))) # (x, y)

        return np.array(points, dtype=np.int32)

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
        FUNGSI GABUNGAN - MULTI FACE VERSION:
        1. Deteksi SEMUA wajah stabil (HOG+SVM Kustom)
        2. Deteksi landmark presisi (LBF) untuk setiap wajah
        3. Aplikasi blush pada semua wajah
        """
        output_frame = frame.copy().astype(np.float32)
        
        # 1. Konversi ke grayscale 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Deteksi SEMUA wajah (Menggunakan fungsi HOG+SVM baru)
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
            # (all_faces sudah dalam format numpy array dari detect_face_robust)
            
            try:
                ok, landmarks_list = self.landmark_detector.fit(gray, all_faces)
            except cv2.error as e:
                # Gagal fit, kembalikan frame asli
                # print(f"Landmark fit error: {e}") # Opsional: untuk debug
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

    # --- (Bagian Jaringan / Socket TIDAK BERUBAH) ---

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
        time.sleep(0.1) # Beri waktu thread untuk berhenti

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
    print("=== Hybrid CV2 Blush Server (Kustom HOG+SVM + Precise LBF) ===") # Judul diubah
    print("📝 Dependencies: pip install opencv-python-contrib scipy numpy scikit-image joblib") # Tambahkan scikit-image & joblib
    print("⚠️  Pastikan 'lbfmodel.yaml' DAN 'face_detector_hog_svm.pkl' ada di folder!")
    print("✨ Features:")
    print("    - Kustom HOG+SVM face detection")
    print("    - LBF Landmark-based cheek positioning")
    
    server = UDPWebcamServer()
    try:
        server.start_server()
    except Exception as e:
        print(f"💥 Unhandled exception: {e}")
        server.stop_server()