"""
Batch Trim
---------------------
Reads metadata.csv (must have columns: filepath, start_sec, end_sec)
and trims every raw clip to just the shooting motion, saving to trimmed/.
"""
import os

import pandas as pd
from moviepy.editor import VideoFileClip


def iter_trim_requests_from_dataframe(metadata):
    for _, row in metadata.iterrows():
        filepath = row.get("filepath")
        if pd.isna(filepath):
            continue

        filepath = str(filepath).strip()
        if not filepath:
            continue

        start_sec = row.get("start_sec")
        end_sec = row.get("end_sec")
        if pd.isna(start_sec) or pd.isna(end_sec):
            print(f"Skipping {filepath}: missing trim range")
            continue

        yield filepath, float(start_sec), float(end_sec)


def iter_trim_requests(metadata_path="metadata.csv"):
    metadata = pd.read_csv(metadata_path)
    yield from iter_trim_requests_from_dataframe(metadata)


def trim_videos(metadata_path="metadata.csv"):
    for filepath, start_sec, end_sec in iter_trim_requests(metadata_path):
        out_path = filepath.replace("raw/", "trimmed/", 1)
        output_dir = os.path.dirname(out_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if os.path.exists(out_path):
            print(f"Skipping (already trimmed): {out_path}")
            continue

        print(f"Trimming {filepath} -> {out_path}")
        clip = VideoFileClip(filepath).subclip(start_sec, end_sec)
        clip.write_videofile(out_path, codec="libx264", audio=False, logger=None)


def main():
    trim_videos()
    print("Batch trim complete.")


if __name__ == "__main__":
    main()