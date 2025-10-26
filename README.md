# Virtual Try-On Blush

Aplikasi virtual try-on blush yang menggunakan deteksi wajah real-time untuk menerapkan efek blush pada pipi secara otomatis. Aplikasi ini terdiri dari klien berbasis Godot dan server UDP Python yang memproses webcam secara real-time.

## Fitur Utama

- **Deteksi Wajah Stabil**: Menggunakan Haar Cascade dan temporal smoothing untuk deteksi wajah yang stabil
- **Deteksi Landmark Presisi**: Menggunakan model LBF (Local Binary Features) untuk deteksi landmark wajah 68 titik
- **Aplikasi Blush Otomatis**: Efek blush diterapkan pada area pipi berdasarkan landmark wajah
- **Streaming Real-Time**: Server UDP mengirim frame yang diproses ke klien Godot
- **Kontrol Dinamis**: Pengaturan warna, intensitas, dan blur blush dapat diubah secara real-time
- **Antarmuka Pengguna**: Menu utama, tutorial, tentang aplikasi, dan kredit

## Persyaratan Sistem

### Klien (Godot)
- Godot Engine 4.3 atau lebih baru
- Sistem operasi: Windows, macOS, atau Linux

### Server (Python)
- Python 3.7+
- OpenCV dengan kontrib (opencv-python-contrib)
- NumPy
- SciPy

## Instalasi

### 1. Kloning Repository
```bash
git clone <repository-url>
cd ETS-PCD-Blush
```

### 2. Instalasi Dependensi Python
```bash
cd Virtual_Try_On_Blush
pip install opencv-python-contrib scipy numpy
```

### 3. Persiapan Model
Pastikan file `lbfmodel.yaml` berada di folder `Virtual_Try_On_Blush/`. File ini berukuran sekitar 54MB dan berisi model landmark 68 titik.

### 4. Jalankan Godot
Buka folder `tubes_ets_pcd` dengan Godot Engine dan jalankan proyek.

## Penggunaan

### Menjalankan Server
```bash
cd Virtual_Try_On_Blush
python udp_webcam_server.py
```

Server akan mulai mendengarkan pada:
- Port utama: 8888 (untuk streaming frame)
- Port kontrol: 8889 (untuk pengaturan blush)

### Menjalankan Klien
1. Buka proyek Godot di folder `tubes_ets_pcd`
2. Jalankan scene utama (`menu_utama.tscn`)
3. Klik tombol untuk memulai try-on blush
4. Webcam akan aktif dan efek blush akan diterapkan secara real-time

### Kontrol Blush
Server menerima perintah UDP untuk mengubah pengaturan:
- **Warna**: `COLOR:R,G,B` (contoh: `COLOR:235,148,146`)
- **Intensitas**: `INTENSITY:0.0-1.0` (contoh: `INTENSITY:0.25`)
- **Blur**: `BLUR:5-50` (contoh: `BLUR:15`)

## Struktur Proyek

```
ETS-PCD-Blush/
├── tubes_ets_pcd/              # Proyek Godot (Klien)
│   ├── Scene/                  # Scene Godot
│   │   ├── menu_utama.tscn     # Menu utama
│   │   ├── about.tscn          # Halaman tentang
│   │   ├── tutor.tscn          # Tutorial
│   │   ├── kredit.tscn         # Kredit
│   │   └── webcam_client*.tscn # Scene webcam
│   ├── Script/                 # Script GDScript
│   │   ├── menu_utama.gd       # Logika menu utama
│   │   ├── button.gd           # Logika tombol
│   │   ├── Sound.gd            # Manajemen suara
│   │   └── webcam_client*.gd   # Klien UDP webcam
│   ├── assets/                 # Aset (gambar, font, suara)
│   └── project.godot           # Konfigurasi proyek Godot
├── Virtual_Try_On_Blush/       # Server Python
│   ├── udp_webcam_server.py   # Server UDP utama
│   └── lbfmodel.yaml           # Model landmark wajah
└── README.md                   # Dokumentasi ini
```

## Cara Kerja

1. **Deteksi Wajah**: Server menggunakan Haar Cascade untuk deteksi wajah awal
2. **Pelacakan Landmark**: Model LBF mendeteksi 68 titik landmark pada wajah
3. **Identifikasi Posisi Pipi**: Titik landmark digunakan untuk menentukan area pipi
4. **Aplikasi Efek**: Mask blush dibuat dan diterapkan pada area pipi dengan blending
5. **Streaming**: Frame yang diproses dikirim ke klien Godot via UDP

## Kontribusi

1. Fork repository
2. Buat branch fitur baru (`git checkout -b fitur-baru`)
3. Commit perubahan (`git commit -am 'Tambah fitur baru'`)
4. Push ke branch (`git push origin fitur-baru`)
5. Buat Pull Request
