#!/usr/bin/env python3

import shutil
from pathlib import Path
import pandas as pd
import logging


def copy_files(source: str, dest: str) -> None:
    """
    Copy files from source to destination.
    If the source file does not exist, log a warning.
    """
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    try:
        if Path(source).exists():
            shutil.copy2(source, dest)
        else:
            logging.warning(f"Source file not found: {source}")
    except Exception as e:
        logging.error(f"Failed to copy {Path(source).name}: {e}")


if __name__ == "__main__":
    file_paths = pd.read_csv("test.csv")
    for index, row in file_paths.iterrows():
        print(
            f"Processing row {index + 1}/{len(file_paths)}: {row['detection_image_path']}"
        )
        if pd.notna(row["cropped_image"]):
            class_source = row["cropped_image"]
            class_dest = row["classification_dest_image"]
            copy_files(class_source, class_dest)
        label = row["label"]
        label_dest = row["destination_labels"]
        image_dest = row["destination_images"]
        image_source = row["detection_image_path"]
        copy_files(image_source, image_dest)
        try:
            Path(label_dest).parent.mkdir(parents=True, exist_ok=True)
            with open(label_dest, "w") as f:
                f.write(str(label))
        except Exception as e:
            print(f"Failed to create label {label_dest}: {e}")
