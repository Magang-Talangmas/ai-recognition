import argparse

from attendance_cv.config import load_config
from attendance_cv.database import AttendanceDB


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("event_id", type=int)
    parser.add_argument(
        "action",
        choices=[
            "confirm",
            "reject",
            "break",
            "temporary_exit",
            "checkout",
            "return_break",
            "return_temporary",
        ],
    )
    args = parser.parse_args()

    config = load_config()
    database = AttendanceDB(
        config.attendance.event_database,
        config.attendance.duplicate_cooldown_seconds,
    )
    database.respond(args.event_id, args.action)
    print(f"Event {args.event_id} updated with action {args.action}")


if __name__ == "__main__":
    main()
