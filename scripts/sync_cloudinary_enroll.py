"""
Sync Employee Photos from Cloudinary & Auto-Generate C Binary Embeddings (embeddings.bin)
Standalone inside ai-recognition-c. Zero external dependencies required.

Features:
- Handles Cloudinary random file names / timestamp suffixes automatically.
- Supported Cloudinary naming structures:
    1. employees/EMP001/any_random_hash.jpg  -> Employee ID: EMP001
    2. employees/EMP001_1712398471.jpg       -> Employee ID: EMP001
    3. employees/EMP001.jpg                  -> Employee ID: EMP001
    4. EMP001_random_filename.jpg            -> Employee ID: EMP001
- Downloads photos to data/enroll/<employee_id>/
- Triggers native C enroll_employee or ArcFace engine to build data/embeddings.bin
"""

import os
import re
import sys
import json
import base64
import struct
import argparse
import subprocess
import importlib.util
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
import urllib.request
import urllib.error

# Ensure Windows stdout handles UTF-8 gracefully
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def sanitize_id(name: str) -> str:
    """Sanitize string to be clean alphanumeric / hyphen ID."""
    return re.sub(r'[\\/*?:"<>|]', "_", name).strip()


def parse_cloudinary_credentials() -> dict[str, str] | None:
    """Parse Cloudinary credentials from environment variables or .env file."""
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip().strip('"').strip("'")

    url = os.environ.get("CLOUDINARY_URL", "").strip()
    if url and url.startswith("cloudinary://"):
        parsed = urlparse(url)
        return {
            "cloud_name": parsed.hostname or "",
            "api_key": parsed.username or "",
            "api_secret": parsed.password or "",
            "folder": os.environ.get("CLOUDINARY_FOLDER", "").strip(),
        }

    cloud_name = os.environ.get("CLOUDINARY_CLOUD_NAME", "").strip()
    api_key = os.environ.get("CLOUDINARY_API_KEY", "").strip()
    api_secret = os.environ.get("CLOUDINARY_API_SECRET", "").strip()
    folder = os.environ.get("CLOUDINARY_FOLDER", "").strip()

    if cloud_name and api_key and api_secret:
        return {
            "cloud_name": cloud_name,
            "api_key": api_key,
            "api_secret": api_secret,
            "folder": folder,
        }

    return None


def extract_employee_id_from_public_id(public_id: str, root_folder: str) -> tuple[str, str]:
    """
    Extracts the clean Employee ID from Cloudinary public_id.
    Handles random timestamps, hashes, or nested paths.
    """
    clean_id = public_id
    if root_folder and clean_id.startswith(root_folder):
        clean_id = clean_id[len(root_folder):].lstrip("/")

    # Subfolder: employees/EMP001/random_photo_id
    if "/" in clean_id:
        parts = clean_id.split("/")
        emp_id = sanitize_id(parts[0])
        file_part = "_".join(parts[1:])
        return emp_id, file_part

    # Underscore suffix: EMP001_1720394857
    if "_" in clean_id:
        emp_id_part, _, file_part = clean_id.rpartition("_")
        return sanitize_id(emp_id_part), file_part

    return sanitize_id(clean_id), "photo"


def http_get_json(url: str, auth_user: str, auth_pass: str) -> dict[str, Any]:
    """Standard library HTTP GET with Basic Auth."""
    req = urllib.request.Request(url)
    credentials = f"{auth_user}:{auth_pass}".encode("utf-8")
    base64_creds = base64.b64encode(credentials).decode("ascii")
    req.add_header("Authorization", f"Basic {base64_creds}")

    with urllib.request.urlopen(req, timeout=15) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP Error {response.status}")
        data = response.read().decode("utf-8")
        return json.loads(data)


def fetch_cloudinary_images(creds: dict[str, str]) -> list[dict[str, Any]]:
    """Fetch all images from Cloudinary Admin API."""
    cloud_name = creds["cloud_name"]
    api_key = creds["api_key"]
    api_secret = creds["api_secret"]
    folder = creds["folder"]

    base_url = f"https://api.cloudinary.com/v1_1/{cloud_name}/resources/image/upload?max_results=500"
    if folder:
        base_url += f"&prefix={urllib.parse.quote(folder)}"
        print(f"≡ƒôí Menghubungi Cloudinary [{cloud_name}] pada folder: '{folder}'...")
    else:
        print(f"≡ƒôí Menghubungi Cloudinary [{cloud_name}] (root storage)...")

    resources = []
    next_cursor = None

    while True:
        url = base_url
        if next_cursor:
            url += f"&next_cursor={urllib.parse.quote(next_cursor)}"

        data = http_get_json(url, api_key, api_secret)
        resources.extend(data.get("resources", []))
        next_cursor = data.get("next_cursor")
        if not next_cursor:
            break

    print(f"≡ƒôÑ Ditemukan {len(resources)} file foto di Cloudinary.")
    return resources


def download_photos(resources: list[dict[str, Any]], root_folder: str, target_dir: Path, force: bool = False) -> tuple[int, int, set[str]]:
    """Download images from Cloudinary into data/enroll/<employee_id>/."""
    target_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    skipped = 0
    updated_employees = set()

    for res in resources:
        public_id = res.get("public_id", "")
        secure_url = res.get("secure_url") or res.get("url")
        file_format = res.get("format", "jpg")

        if not public_id or not secure_url:
            continue

        emp_id, file_suffix = extract_employee_id_from_public_id(public_id, root_folder)
        emp_folder = target_dir / emp_id
        emp_folder.mkdir(parents=True, exist_ok=True)

        local_file = emp_folder / f"{file_suffix}.{file_format}"

        if local_file.exists() and not force:
            skipped += 1
            continue

        try:
            req = urllib.request.Request(secure_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                if resp.status == 200:
                    with open(local_file, "wb") as f:
                        f.write(resp.read())
                    downloaded += 1
                    updated_employees.add(emp_id)
                    print(f"  Γ¼ç∩╕Å [DOWNLOAD] Employee: '{emp_id}' <- {local_file.name}")
        except Exception as e:
            print(f"  ΓÜá∩╕Å Gagal download {secure_url}: {e}")

    return downloaded, skipped, updated_employees


def run_c_enrollment(enroll_dir: Path, bin_path: Path) -> None:
    """Run native C enrollment CLI for each employee folder."""
    exe_name = "enroll_employee.exe" if sys.platform == "win32" else "./enroll_employee"
    exe_path = Path(exe_name)

    if not exe_path.exists():
        print(f"\nΓä╣∩╕Å Binary '{exe_name}' belum di-build.")
        print("   Jalankan 'build_and_test.bat' terlebih dahulu untuk mengompilasi.")
        return

    print(f"\n≡ƒºá Menjalankan Native C Enrollment Engine ({exe_name})...")
    for emp_dir in sorted(enroll_dir.iterdir()):
        if not emp_dir.is_dir():
            continue
        emp_id = emp_dir.name
        cmd = [str(exe_path), emp_id, str(emp_dir), str(bin_path)]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                print(f"  Γ£à [C ENROLLED] '{emp_id}'")
            else:
                print(f"  ΓÜá∩╕Å [WARNING] '{emp_id}': {res.stderr.strip() or res.stdout.strip()}")
        except Exception as err:
            print(f"  Γ¥î Error running {exe_name} for '{emp_id}': {err}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync Employee Photos from Cloudinary & Auto-Generate C Binary Embeddings (embeddings.bin)"
    )
    parser.add_argument("--force", action="store_true", help="Paksa download ulang semua foto")
    parser.add_argument("--bin-out", default="data/embeddings.bin", help="Path file binary C output")
    parser.add_argument("--enroll-dir", default="data/enroll", help="Folder lokal tempat foto disimpan")
    args = parser.parse_args()

    print("=================================================================")
    print("   AI-RECOGNITION-C : CLOUDINARY SYNC & BINARY ENROLLMENT        ")
    print("=================================================================")

    creds = parse_cloudinary_credentials()
    if not creds:
        print("Γ¥î Kredensial Cloudinary belum diatur di .env!")
        print("   Silakan atur: CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET")
        sys.exit(1)

    target_enroll = Path(args.enroll_dir)
    target_bin = Path(args.bin_out)

    try:
        resources = fetch_cloudinary_images(creds)
        downloaded, skipped, updated = download_photos(
            resources=resources,
            root_folder=creds["folder"],
            target_dir=target_enroll,
            force=args.force
        )
        print(f"≡ƒôè Ringkasan: {downloaded} foto baru diunduh, {skipped} foto sudah ada.")

        # Dynamically check if python insightface is available, else use native C runner
        has_insightface = importlib.util.find_spec("insightface") is not None
        has_cv2 = importlib.util.find_spec("cv2") is not None

        if has_insightface and has_cv2:
            print("\n≡ƒºá Menjalankan model InsightFace Python untuk ekstraksi embedding...")
            cv = importlib.import_module("cv2")
            insightface_mod = importlib.import_module("insightface.app")
            np = importlib.import_module("numpy")

            FaceAnalysis = getattr(insightface_mod, "FaceAnalysis")
            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=0, det_size=(640, 640))

            emp_ids = []
            templates = []

            for emp_dir in sorted(target_enroll.iterdir()):
                if not emp_dir.is_dir():
                    continue

                emp_vectors = []
                for p in emp_dir.glob("*"):
                    if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                        img = cv.imread(str(p))
                        if img is None:
                            continue
                        faces = app.get(img)
                        if faces and len(faces) > 0:
                            best_face = max(faces, key=lambda f: f.det_score)
                            emp_vectors.append(best_face.embedding)

                if emp_vectors:
                    mean_vec = np.mean(emp_vectors, axis=0)
                    mean_vec = mean_vec / np.linalg.norm(mean_vec)
                    emp_ids.append(emp_dir.name)
                    templates.append(mean_vec)
                    print(f"  Γ£à [ENROLLED] '{emp_dir.name}' ({len(emp_vectors)} foto valid)")

            if emp_ids:
                # Save binary format FACES1
                target_bin.parent.mkdir(parents=True, exist_ok=True)
                with open(target_bin, "wb") as f:
                    f.write(b"FACES1\x00\x00")
                    f.write(struct.pack("<ii", len(emp_ids), 512))
                    for i in range(len(emp_ids)):
                        emp_id_bytes = emp_ids[i].encode("utf-8")[:127].ljust(128, b"\x00")
                        f.write(emp_id_bytes)
                        vec = templates[i].astype(np.float32)
                        f.write(vec.tobytes())
                print(f"\n≡ƒÆ╛ [C Binary Ready] Berhasil membuat '{target_bin}' ({len(emp_ids)} karyawan).")
        else:
            # Native C engine runner fallback
            run_c_enrollment(target_enroll, target_bin)

    except Exception as err:
        print(f"Γ¥î Terjadi kesalahan: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
