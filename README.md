# AI-Recognition Attendance Engine (C Edition)

Proyek sistem absensi dan pengenalan wajah real-time berbasis C (C11/C99) yang dirancang untuk performa tinggi, efisiensi memori, dan latensi rendah pada kamera RTSP/USB.

Repository GitHub: [https://github.com/Magang-Talangmas/ai-recognition-c.git](https://github.com/Magang-Talangmas/ai-recognition-c.git)

---

## 🌟 Fitur Utama (Features)

1. **High Performance & Low Latency**: Implementasi native C11 tanpa overhead Python runtime / GIL.
2. **Modular Architecture**:
   - `config`: Parser konfigurasi YAML / JSON dengan dukungan ekspansi variabel lingkungan (`.env` dan `${VAR}`).
   - `database`: SQLite3 C interface untuk pencatatan event lokal berstatus `PENDING_CONFIRMATION` / `CONFIRMED` / `REJECTED` dengan proteksi jeda duplikasi (*duplicate cooldown*).
   - `face_engine`: Deteksi wajah SCRFD, kalkulasi skor blur Laplacian variance, penyelarasan 5-titik landmark (*Umeyama Affine Transform 112x112*), dan ekstraksi embedding fitur 512-D ArcFace.
   - `matcher`: Normalisasi vektor L2 in-place, pencocokan Cosine Similarity berkecepatan tinggi, dan margin filter Top-1 vs Top-2.
   - `vision_utils`: Pelacakan centroid wajah (`CentroidTracker`), sliding window majority voting queue, deteksi arah penyeberangan garis virtual (`ENTER` / `EXIT`), dan timestamp UTC ISO8601.
   - `api_dispatcher`: Worker thread latar belakang (*asynchronous HTTP POST dispatch*) non-blocking dengan payload JSON `CHECK_IN` dan header autentikasi `x-api-key`.
3. **Cross-Platform**: Mendukung Windows (WinHTTP / Win32 Threads) dan Linux / macOS (POSIX Threads / libcurl).

---

## 📂 Struktur Direktori

```text
camera-c/
├── CMakeLists.txt              # Konfigurasi Build CMake
├── config.yaml                 # File konfigurasi utama
├── .env.example                # Contoh variabel environment
├── .gitignore                  # File git ignore
├── README.md                   # Dokumentasi lengkap
├── include/                    # Header file C
│   ├── attendance/
│   │   ├── config.h
│   │   ├── database.h
│   │   ├── face_engine.h
│   │   ├── matcher.h
│   │   ├── vision_utils.h
│   │   └── api_dispatcher.h
│   └── third_party/
│       └── cJSON.h
├── src/                        # Source code implementasi C
│   ├── config.c
│   ├── database.c
│   ├── face_engine.c
│   ├── matcher.c
│   ├── vision_utils.c
│   ├── api_dispatcher.c
│   ├── run_camera.c            # Entry point streaming kamera
│   ├── enroll_employee.c       # CLI pendaftaran wajah karyawan
│   ├── respond_event.c         # CLI konfirmasi / respons event absensi
│   ├── export_embeddings.c     # CLI inspeksi database embedding
│   └── third_party/
│       └── cJSON.c
└── tests/                      # Unit & Integration Tests
    └── test_all.c
```

---

## 🛠️ Panduan Build & Kompilasi

### Prasyarat
- CMake 3.16+
- C Compiler (GCC / Clang / MSVC)
- SQLite3 Development Library
- *(Opsional)* OpenCV C API, ONNX Runtime C API / OpenVINO C API

### Kompilasi Menggunakan CMake:
```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

### Menjalankan Unit & Integration Tests:
```bash
ctest --output-on-failure
# atau langsung jalankan binary test:
./test_all
```

---

## 🚀 Penggunaan (CLI Commands)

### 1. Menjalankan Stream Kamera Real-Time
```bash
./run_camera config.yaml
```

### 2. Mendaftarkan Wajah Karyawan (Enrollment)
```bash
./enroll_employee <employee_id> <photo_folder> [data/embeddings.bin]
# Contoh:
./enroll_employee EMP-001 photos/emp001/ data/embeddings.bin
```

### 3. Mengonfirmasi atau Menolak Event Absensi
```bash
# Menampilkan daftar 50 event terakhir:
./respond_event --list

# Mengonfirmasi event:
./respond_event 101 confirm

# Menolak event (Bukan saya / Not Me):
./respond_event 101 reject
```

### 4. Memeriksa Database Template Embedding
```bash
./export_embeddings data/embeddings.bin
```

---

## 📜 Lisensi & Kontributor
- **Organisasi**: PT. Talangmas Teknologi Indonesia / Magang Talangmas
- **Lisensi**: MIT
