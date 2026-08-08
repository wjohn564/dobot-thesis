import csv
from robot_sorting import config as cfg

# Define the columns of calibration_points.csv
FIELDNAMES = [
    "point_id",
    "u",
    "v",
    "robot_x",
    "robot_y",
    "robot_z",
    "colour",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "area",
    "image_name",
    "preview_name",
    "camera_time",
    "robot_time",
    "notes",
]


def read_rows():
    """Read the calibration points CSV."""
    if not cfg.CALIBRATION_POINTS_CSV.exists():
        return []

    with open(
            cfg.CALIBRATION_POINTS_CSV,
            "r",
            newline="",
            encoding="utf-8",
    ) as csv_file:
        return list(csv.DictReader(csv_file))


def write_rows(rows):
    """Write all calibration points to the CSV."""
    with open(
            cfg.CALIBRATION_POINTS_CSV,
            "w",
            newline="",
            encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=FIELDNAMES,
        )

        writer.writeheader()

        for row in rows:
            clean_row = {
                field: row.get(field, "")
                for field in FIELDNAMES
            }

            writer.writerow(clean_row)


def next_point_id():
    """Return the next calibration point ID."""
    highest_number = 0

    for row in read_rows():
        point_id = row.get(
            "point_id",
            "",
        ).strip()

        if not point_id.startswith("p"):
            continue

        try:
            number = int(point_id[1:])
            highest_number = max(
                highest_number,
                number,
            )
        except ValueError:
            continue

    return f"p{highest_number + 1:02d}"


def find_next_unfilled_row(rows):
    """Return the next calibration point without robot coordinates."""
    for row in rows:
        if not row.get("robot_x"):
            return row

    return None
