"""
Sync Employee Photos from Cloudinary & Auto-Enroll Face Embeddings.

Usage:
  python sync_and_enroll.py              # Sync photos from Cloudinary + generate embeddings.npz
  python sync_and_enroll.py --sync-only  # Only download/update photos from Cloudinary
  python sync_and_enroll.py --enroll-only# Only generate embeddings.npz from local folder
  python sync_and_enroll.py --force      # Re-download all photos even if they exist locally
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# Ensure Windows stdout handles UTF-8 gracefully
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2 as cv
import numpy as np
import requests

from attendance_cv.config import load_config
from attendance_cv.face_engine import FaceEngine


def parse_cloudinary_credentials() -> dict[str, str] | None:
    """Parse Cloudinary credentials from CLOUDINARY_URL or individual env vars."""
    url = os.environ.get("CLOUDINARY_URL", "").strip()
    if url and url.startswith("cloudinary://"):
        parsed = urlparse(url)
        return {
            "cloud_name": parsed.hostname or "",
            "api_key": parsed.username or "",
            "api_secret": parsed.password or "",
            "folder": os.environ.get("CLOUDINARY_FOLDER", "employees").strip(),
        }

    cloud_name = os.environ.get("CLOUDINARY_CLOUD_NAME", "").strip()
    api_key = os.environ.get("CLOUDINARY_API_KEY", "").strip()
    api_secret = os.environ.get("CLOUDINARY_API_SECRET", "").strip()
    folder = os.environ.get("CLOUDINARY_FOLDER", "employees").strip()

    if cloud_name and api_key and api_secret:
        return {
            "cloud_name": cloud_name,
            "api_key": api_key,
            "api_secret": api_secret,
            "folder": folder,
        }

    return None


def fetch_cloudinary_resources(creds: dict[str, str]) -> list[dict[str, Any]]:
    """Fetch all image resources in the given folder prefix from Cloudinary Admin API."""
    cloud_name = creds["cloud_name"]
    api_key = creds["api_key"]
    api_secret = creds["api_secret"]
    folder = creds["folder"]

    api_url = f"https://api.cloudinary.com/v1_1/{cloud_name}/resources/image/upload"
    params: dict[str, Any] = {
        "prefix": folder,
        "max_results": 500,
    }

    all_resources: list[dict[str, Any]] = []
    next_cursor = None

    print(f"📡 Menghubungi Cloudinary ({cloud_name}) pada folder: '{folder}'...")

    while True:
        if next_cursor:
            params["next_cursor"] = next_cursor

        response = requests.get(
            api_url,
            params=params,
            auth=(api_key, api_secret),
            timeout=15.0,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Cloudinary API Error [{response.status_code}]: {response.text}"
            )

        data = response.json()
        resources = data.get("resources", [])
        all_resources.extend(resources)

        next_cursor = data.get("next_cursor")
        if not next_cursor:
            break

    print(f"📥 Ditemukan {len(all_resources)} file foto di Cloudinary.")
    return all_resources


def extract_employee_id_from_public_id(public_id: str, root_folder: str) -> tuple[str, str]:
    """
    Extract (employee_id, filename) from Cloudinary public_id.
    Example: 'employees/EMP001/face1' -> ('EMP001', 'face1')
             'employees/akmalShaumNadzirin/01' -> ('akmalShaumNadzirin', '01')
             'EMP001_photo1' -> ('EMP001', 'photo1')
    """
    clean_id = public_id.removeprefix(root_folder).strip("/")
    parts = clean_id.split("/")

    if len(parts) >= 2:
        employee_id = parts[0]
        filename = "_".join(parts[1:])
    else:
        # Fallback if flat structure: e.g. EMP001_photo1
        match = re.match(r"^([^_]+)_(.+)$", clean_id)
        if match:
            employee_id, filename = match.groups()
        else:
            employee_id = clean_id
            filename = "default"

    return employee_id, filename


def download_cloudinary_photos(
    resources: list[dict[str, Any]],
    root_folder: str,
    target_dir: Path,
    force: bool = False,
) -> int:
    """Download images from Cloudinary and organize into target_dir/<employee_id>/."""
    target_dir.mkdir(parents=True, exist_ok=True)
    downloaded_count = 0

    for res in resources:
        public_id = res.get("public_id", "")
        secure_url = res.get("secure_url") or res.get("url")
        file_format = res.get("format", "jpg")

        if not public_id or not secure_url:
            continue

        employee_id, filename = extract_employee_id_from_public_id(public_id, root_folder)
        emp_dir = target_dir / employee_id
        emp_dir.mkdir(parents=True, exist_ok=True)

        local_file = emp_dir / f"{filename}.{file_format}"

        if local_file.exists() and not force:
            continue

        try:
            img_resp = requests.get(secure_url, timeout=15.0)
            if img_resp.status_code == 200:
                with open(local_file, "wb") as f:
                    f.write(img_resp.content)
                downloaded_count += 1
                print(f"  ⬇️ Downloaded: {employee_id}/{local_file.name}")
            else:
                print(f"  ⚠️ Gagal download {secure_url}: HTTP {img_resp.status_code}")
        except Exception as err:
            print(f"  ❌ Error downloading {secure_url}: {err}")

    return downloaded_count


def build_face_embeddings(
    enroll_root: Path,
    output_path: Path,
    face_config: Any,
) -> int:
    """Build ArcFace embedding templates from local enroll folder and save as .npz."""
    print(f"\n🧠 Membangun template embedding wajah dari '{enroll_root}'...")
    engine = FaceEngine(face_config)

    employee_ids: list[str] = []
    templates: list[np.ndarray] = []

    if not enroll_root.exists():
        raise FileNotFoundError(f"Folder '{enroll_root}' tidak ditemukan!")

    employee_dirs = [d for d in sorted(enroll_root.iterdir()) if d.is_dir()]
    if not employee_dirs:
        raise RuntimeError(f"Tidak ada subfolder employee di '{enroll_root}'")

    for employee_dir in employee_dirs:
        images: list[np.ndarray] = []
        for image_path in sorted(employee_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                continue
            image = cv.imread(str(image_path))
            if image is not None:
                images.append(image)

        if not images:
            print(f"  ⚠️ SKIP {employee_dir.name}: tidak ada file foto valid")
            continue

        try:
            template, valid_count = engine.build_template(images)
            employee_ids.append(employee_dir.name)
            templates.append(template)
            print(f"  ✅ OK [{employee_dir.name}]: {valid_count}/{len(images)} foto valid")
        except ValueError as exc:
            print(f"  ⚠️ SKIP [{employee_dir.name}]: {exc}")
            continue

    if not templates:
        raise RuntimeError("Gagal membuat template: tidak ada employee valid yang terdeteksi.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        employee_ids=np.asarray(employee_ids),
        templates=np.stack(templates),
    )
    print(f"\n💾 Berhasil menyimpan {len(employee_ids)} embedding employee ke: '{output_path}'")
    return len(employee_ids)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync Employee Photos from Cloudinary & Auto-Enroll Face Embeddings."
    )
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="Hanya download foto dari Cloudinary tanpa membuat embedding",
    )
    parser.add_argument(
        "--enroll-only",
        action="store_true",
        help="Hanya membuat embedding dari folder lokal tanpa kontak ke Cloudinary",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Paksa download ulang semua foto dari Cloudinary",
    )
    args = parser.parse_args()

    # Load system configuration
    config = load_config()

    enroll_dir = Path(os.environ.get("ENROLL_DATA_PATH", "data/enroll"))
    output_npz = Path(os.environ.get("EMBEDDING_FILE_PATH", config.attendance.embedding_file))

    start_time = time.time()
    print("=" * 65)
    print("🚀 TALANGMAS ATTENDANCE - CLOUD SYNC & FACE ENROLLMENT")
    print("=" * 65)

    if not args.enroll_only:
        creds = parse_cloudinary_credentials()
        if not creds:
            print(
                "⚠️ Kredensial Cloudinary belum diatur di .env "
                "(CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET)."
            )
            print("➡️ Melanjutkan proses enrollment dari file lokal yang ada...\n")
        else:
            try:
                resources = fetch_cloudinary_resources(creds)
                downloaded = download_cloudinary_photos(
                    resources=resources,
                    root_folder=creds["folder"],
                    target_dir=enroll_dir,
                    force=args.force,
                )
                print(f"✅ Sync Selesai: {downloaded} foto baru/diperbarui.")
            except Exception as err:
                print(f"❌ Terjadi kesalahan saat sync Cloudinary: {err}")
                print("➡️ Mencoba melanjutkan dengan data lokal yang sudah ada...")

    if not args.sync_only:
        try:
            total_enrolled = build_face_embeddings(
                enroll_root=enroll_dir,
                output_path=output_npz,
                face_config=config.face,
            )
        except Exception as err:
            print(f"❌ Error saat enrollment wajah: {err}")
            sys.exit(1)

    duration = time.time() - start_time
    print("=" * 65)
    print(f"🎉 Selesai dalam {duration:.2f} detik!")
    print("=" * 65)


if __name__ == "__main__":
    main()
