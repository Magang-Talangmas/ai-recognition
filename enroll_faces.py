from pathlib import Path

import cv2 as cv
import numpy as np

from attendance_cv.config import load_config
from attendance_cv.face_engine import FaceEngine


def main() -> None:
    config = load_config()
    engine = FaceEngine(config.face)
    enroll_root = Path("data/enroll")

    employee_ids: list[str] = []
    templates: list[np.ndarray] = []

    if not enroll_root.exists():
        raise FileNotFoundError(
            "Create data/enroll/<employee_id>/ with at least 3 photos"
        )

    for employee_dir in sorted(enroll_root.iterdir()):
        if not employee_dir.is_dir():
            continue

        images = []
        for image_path in sorted(employee_dir.iterdir()):
            image = cv.imread(str(image_path))
            if image is not None:
                images.append(image)

        try:
            template, valid_count = engine.build_template(images)
        except ValueError as exc:
            print(f"SKIP {employee_dir.name}: {exc}")
            continue

        employee_ids.append(employee_dir.name)
        templates.append(template)
        print(
            f"OK {employee_dir.name}: "
            f"{valid_count} valid photo(s)"
        )

    if not templates:
        raise RuntimeError("No employee template was created")

    output = Path(config.attendance.embedding_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        employee_ids=np.asarray(employee_ids),
        templates=np.stack(templates),
    )
    print(f"Saved {len(employee_ids)} employee template(s) to {output}")


if __name__ == "__main__":
    main()
