#!/usr/bin/env python

import os
from pathlib import Path
from PIL import Image, ExifTags
from datetime import datetime
import collections
import sys


def get_exif_datetime(image_path):
    """
    Extracts the original datetime from an image's EXIF data.
    Prioritizes 'DateTimeOriginal', then 'DateTimeDigitized', then 'DateTime'.
    Returns a datetime object or None if no suitable tag is found.
    """
    try:
        img = Image.open(image_path)
        exif_data = (
            img._getexif()
        )  # _getexif() is generally reliable, getexif() is newer in Pillow

        if exif_data is None:
            return None

        # Map EXIF tag IDs to their names
        exif_tags_map = {v: k for k, v in ExifTags.TAGS.items()}

        date_time_original_tag = exif_tags_map.get("DateTimeOriginal")
        date_time_digitized_tag = exif_tags_map.get("DateTimeDigitized")
        date_time_tag = exif_tags_map.get("DateTime")

        datetime_str = None
        if date_time_original_tag and date_time_original_tag in exif_data:
            datetime_str = exif_data[date_time_original_tag]
        elif date_time_digitized_tag and date_time_digitized_tag in exif_data:
            datetime_str = exif_data[date_time_digitized_tag]
        elif date_time_tag and date_time_tag in exif_data:
            datetime_str = exif_data[date_time_tag]

        if datetime_str:
            # EXIF datetime format is "YYYY:MM:DD HH:MM:SS"
            return datetime.strptime(datetime_str, "%Y:%m:%d %H:%M:%S")
        else:
            return None

    except Exception as e:
        print(f"Error reading EXIF data from {image_path}: {e}")
        return None


def rename_images_by_exif_and_subdir(root_dir):
    """
    Renames image files based on EXIF datetime and sub-subdirectory.

    Args:
        root_dir (str): The root directory to start scanning from (e.g., "2025_01_WCAM_originals").
    """
    root_path = Path(root_dir)
    if not root_path.is_dir():
        print(f"Error: Root directory '{root_dir}' not found.")
        return

    # Use a dictionary to store counts for duplicate timestamps
    timestamp_counts = collections.defaultdict(lambda: collections.defaultdict(int))

    # Store pending renames to avoid issues with modifying the directory structure while iterating
    pending_renames = []

    print(f"Scanning directory: {root_dir}")
    # Walk through the directory tree
    for dirpath, dirnames, filenames in os.walk(root_path):
        current_dir_path = Path(dirpath)

        # Check if the current directory is at the level of WCAMxx/100RECNX
        # We need to find the "WCAMxx" part, which is the direct parent of 100RECNX
        # and has "2025_01_WCAM_originals" as its grandparent.

        # Heuristic: Check if the current directory name is '100RECNX', '101RECNX', etc.
        # And its parent is a 'WCAMxx' directory.
        if current_dir_path.name.find("RECNX"):
            print(current_dir_path)
            wcam_dir = current_dir_path.parent.name  # e.g., WCAM01, WCAM13
            # Extract the two digits from WCAMxx
            try:
                wcam_digits = wcam_dir[-2:]  # Assuming WCAMxx
                if not wcam_digits.isdigit():
                    print(
                        f"Skipping {current_dir_path}: Could not extract WCAM digits from '{wcam_dir}'"
                    )
                    continue
            except IndexError:
                print(
                    f"Skipping {current_dir_path}: WCAM directory name '{wcam_dir}' is too short."
                )
                continue

            print(f"Processing images in: {current_dir_path} (WCAM: {wcam_digits})")

            for filename in filenames:
                if filename.lower().endswith((".jpg", ".jpeg")):
                    original_file_path = current_dir_path / filename

                    exif_dt = get_exif_datetime(original_file_path)

                    if exif_dt:
                        # Format: ddmmyyyy-hhmmss
                        timestamp_str = exif_dt.strftime("%d%m%Y-%H%M%S")

                        # Generate base new name
                        base_new_name = f"{timestamp_str}-WCAM{wcam_digits}"

                        # Handle duplicates for the same timestamp within the same WCAM directory
                        # We need to ensure uniqueness across all images for the same timestamp and WCAM
                        # For simplicity, we'll index globally for a given timestamp+WCAM combination

                        # Increment counter for this specific timestamp and WCAM
                        timestamp_counts[wcam_digits][timestamp_str] += 1
                        current_index = timestamp_counts[wcam_digits][timestamp_str]

                        if current_index > 1:
                            new_filename = f"{base_new_name}-idx{current_index:02d}.JPG"  # :02d for two digits (e.g., 01, 02)
                        else:
                            new_filename = f"{base_new_name}.JPG"

                        new_file_path = original_file_path.parent / new_filename

                        # Add to pending renames
                        pending_renames.append((original_file_path, new_file_path))
                    else:
                        print(
                            f"Could not get EXIF datetime for: {original_file_path}. Skipping."
                        )
        else:
            # If not a recognized "100RECNX" type directory, just continue deeper
            pass

    # Perform renames after all files have been scanned
    if not pending_renames:
        print("No images found to rename or no EXIF data available.")
        return

    print("\n--- Proposed Renames ---")
    for original, new in pending_renames:
        print(f"'{original.name}' -> '{new.name}'")

    for original, new in pending_renames:
        try:
            if original != new:  # Only rename if the name actually changes
                os.rename(original, new)
                print(f"Renamed: {original.name} to {new.name}")
            else:
                print(f"Skipped (no change): {original.name}")
        except OSError as e:
            print(f"Error renaming {original.name} to {new.name}: {e}")
    print("\nRenaming complete.")


if __name__ == "__main__":
    root_directory = sys.argv[1]
    print(root_directory)
    rename_images_by_exif_and_subdir(root_directory)
