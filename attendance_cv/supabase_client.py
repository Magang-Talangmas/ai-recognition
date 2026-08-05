import os
import json
import urllib.request
import urllib.error
from datetime import datetime, timezone

class SupabaseClient:
    def __init__(self, url=None, key=None, bucket=None):
        self.url = (url or os.environ.get("SUPABASE_URL", "")).rstrip("/")
        self.key = key or os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
        self.bucket = bucket or os.environ.get("SUPABASE_STORAGE_BUCKET", "recognition")
        
    def is_configured(self) -> bool:
        return bool(self.url and self.key)

    def insert_attendance_event(self, employee_id: str, similarity: float, camera_id: str = "main-entrance", event_type: str = "CHECK_IN", image_url: str = None) -> dict | None:
        """Insert check-in event into Supabase PostgreSQL attendance_events table via PostgREST."""
        if not self.is_configured():
            print("[Supabase] Notice: Credentials not fully set, skipping DB insert.")
            return None
        
        endpoint = f"{self.url}/rest/v1/attendance_events"
        payload = {
            "employee_id": employee_id,
            "similarity": float(similarity),
            "camera_id": camera_id,
            "event_type": event_type,
            "detected_at": datetime.now(timezone.utc).isoformat(),
            "status": "CONFIRMED"
        }
        if image_url:
            payload["image_url"] = image_url

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(endpoint, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("apikey", self.key)
        req.add_header("Authorization", f"Bearer {self.key}")
        req.add_header("Prefer", "return=representation")

        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                resp_bytes = resp.read()
                result = json.loads(resp_bytes.decode("utf-8")) if resp_bytes else {}
                print(f"[Supabase DB SUCCESS] Attendance event logged for '{employee_id}' (sim: {similarity:.2f})")
                return result
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode('utf-8')
            print(f"[Supabase DB HTTP {e.code}] Failed to log event: {err_msg}")
        except Exception as e:
            print(f"[Supabase DB ERROR] {e}")
        return None

    def upload_snapshot(self, image_bytes: bytes, filename: str) -> str | None:
        """Upload snapshot JPEG/PNG to Supabase Storage bucket ('recognition')."""
        if not self.is_configured() or not image_bytes:
            return None
        
        storage_path = f"snapshots/{filename}"
        endpoint = f"{self.url}/storage/v1/object/{self.bucket}/{storage_path}"
        
        req = urllib.request.Request(endpoint, data=image_bytes, method="POST")
        req.add_header("Content-Type", "image/jpeg")
        req.add_header("apikey", self.key)
        req.add_header("Authorization", f"Bearer {self.key}")
        req.add_header("x-upsert", "true")

        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                public_url = f"{self.url}/storage/v1/object/public/{self.bucket}/{storage_path}"
                print(f"[Supabase Storage] Uploaded snapshot: {public_url}")
                return public_url
        except Exception as e:
            print(f"[Supabase Storage ERROR] Upload failed: {e}")
            return None
