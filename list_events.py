from attendance_cv.config import load_config
from attendance_cv.database import AttendanceDB


def main() -> None:
    config = load_config()
    database = AttendanceDB(
        config.attendance.event_database,
        config.attendance.duplicate_cooldown_seconds,
    )

    rows = database.list_events()
    if not rows:
        print("No event found")
        return

    for row in rows:
        print(
            f"#{row['id']} "
            f"employee={row['employee_id']} "
            f"type={row['event_type']} "
            f"direction={row['direction']} "
            f"status={row['status']} "
            f"score={row['similarity']:.3f} "
            f"time={row['detected_at']}"
        )


if __name__ == "__main__":
    main()
