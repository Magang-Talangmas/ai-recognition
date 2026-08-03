# Python-only SCRFD Attendance Starter

Starter ini hanya berisi bagian Python/computer vision.

Tidak ada:

- admin website;
- mobile application;
- FastAPI;
- authentication;
- PostgreSQL;
- Redis.

Pipeline:

```text
RTSP / webcam
→ YOLOv10 person detection
→ ByteTrack
→ SCRFD face detection
→ face embedding
→ employee matching
→ multi-frame voting
→ virtual-line crossing
→ SQLite attendance event
```

## Struktur

```text
python-only-scrfd-attendance-starter/
├── attendance_cv/
│   ├── config.py
│   ├── database.py
│   ├── face_engine.py
│   ├── matcher.py
│   └── vision_utils.py
├── data/
│   └── enroll/
├── models/
├── enroll_faces.py
├── run_camera.py
├── confirm_event.py
├── list_events.py
├── config.yaml
└── requirements.txt
```

## Instalasi

Gunakan Python 3.11.

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Model

### Person detector

Letakkan YOLOv10 checkpoint di:

```text
models/yolov10n.pt
```

Atau ubah `person_detection.model_path` di `config.yaml`.

### SCRFD dan recognition

Starter menggunakan InsightFace `FaceAnalysis` dengan:

```yaml
face:
  model_pack: buffalo_l
```

Model biasanya akan disimpan di:

```text
.insightface/models/buffalo_l/
```

Model pack tersebut umumnya berisi SCRFD detector dan recognition model.

Periksa lisensi model sebelum deployment nyata.

## Enrollment

Buat folder per employee:

```text
data/enroll/E001/01.jpg
data/enroll/E001/02.jpg
data/enroll/E001/03.jpg

data/enroll/E002/01.jpg
data/enroll/E002/02.jpg
data/enroll/E002/03.jpg
```

Minimal tiga foto valid per employee.

Jalankan:

```bash
python enroll_faces.py
```

Output:

```text
data/embeddings.npz
```

## Menjalankan webcam

Di `config.yaml`:

```yaml
camera:
  source: "0"
```

Kemudian:

```bash
python run_camera.py
```

## Menjalankan RTSP CCTV

```yaml
camera:
  source: "rtsp://username:password@camera-ip/stream"
```

Kemudian:

```bash
python run_camera.py
```

Tekan `q` untuk keluar jika preview aktif.

## Event

Event disimpan di:

```text
data/attendance.db
```

Status awal event:

```text
PENDING_CONFIRMATION
```

Lihat event:

```bash
python list_events.py
```

Simulasi employee confirm:

```bash
python confirm_event.py <EVENT_ID> confirm
```

Simulasi employee reject:

```bash
python confirm_event.py <EVENT_ID> reject
```

Untuk event keluar, pilihan:

```bash
python confirm_event.py <EVENT_ID> break
python confirm_event.py <EVENT_ID> temporary_exit
python confirm_event.py <EVENT_ID> checkout
```

## Catatan penting

`allow_insecure_no_liveness: true` hanya untuk development.

Sebelum production:

- tambahkan presentation attack detection/liveness;
- kalibrasi threshold pada data kantor sendiri;
- uji kamera dan pencahayaan aktual;
- enkripsi template wajah;
- jangan menggunakan raw event langsung untuk payroll;
- hubungkan event Python ini ke backend/mobile pada tahap berikutnya.
