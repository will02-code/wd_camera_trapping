#!/usr/bin/env python3
from ultralytics import YOLO
import os
import cv2
import pandas as pd
from PIL import Image
import re
from megadetector.detection import run_detector
from megadetector.visualization.visualization_utils import draw_bounding_boxes_on_file
import yaml
from datetime import datetime


def load_config(config_path="AI_identification/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def crop_image(img, normalized_coords):
    try:
        x_min = normalized_coords[0][0].int().item()
        y_min = normalized_coords[0][1].int().item()
        x_max = normalized_coords[0][2].int().item()
        y_max = normalized_coords[0][3].int().item()

        cropped_img = img[y_min:y_max, x_min:x_max]
        return cropped_img

    except ValueError:
        print(
            f"Error: Invalid normalized coordinates format: {normalized_coords}. Expected 'x y w h'."
        )
        return None


if __name__ == "__main__":
    config = load_config()

    night_detection = YOLO(config["models"]["night_detection"])

    night_classification = YOLO(config["models"]["night_classification"])
    day_detection = YOLO(config["models"]["day_detection"])
    day_classification = YOLO(config["models"]["day_classification"])
    megadetector = run_detector.load_detector(config["models"]["megadetector_version"])

    wd = os.getcwd()

    all_detections = []
    all_images = []
    digivol_output = []
    base_dir = config["paths"]["base_directory"]

    target_dirs = config["paths"]["target_dirs"]
    for directory in target_dirs:
        image_source_directory = f"{wd}/{base_dir}/{directory}"
        image_training_output = (
            f"{image_source_directory}/{config['output_folders']['output_folder']}"
        )
        image_paths = []
        for dirpath, dirnames, filenames in os.walk(image_source_directory):
            for filename in filenames:
                if filename.endswith(".JPG"):
                    image_path = os.path.join(dirpath, filename)
                    image_paths.append(image_path)
        for image_path in image_paths:
            # try:
            print(f"Processing image: {image_path}")
            # get time of day from image
            image = cv2.imread(image_path)
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hue = hsv_image[:, :, 0].mean()  # H channel represents hue
            # Adjust the thresholds as needed
            if hue > config["thresholds"]["hue_day_threshold"]:
                model_required = "day"
            else:
                model_required = "night"

            print(f"Image is a {model_required} image")

            if model_required == "night":
                detection_model = night_detection
                model = night_classification
            else:
                detection_model = day_detection
                model = day_classification
            camera = re.search(r"WCAM(\d+)", image_path)[0]
            detections = detection_model.predict(
                image,
                save=False,  # I don't save anything here because I want to sort into folders
                save_txt=False,
                save_conf=False,
                iou=0.5,
            )
            img_datetime = (
                Image.open(image_path)
                ._getexif()[36867]
                .replace(" ", "_")
                .replace(":", "_")
            )
            above = [
                d
                for d in detections[0].boxes
                if (d.conf > config["thresholds"]["ultralytics_confidence"])
            ]
            if not above:
                md_detections = megadetector.generate_detections_one_image(image)

                md_detections_filtered = [
                    d
                    for d in md_detections["detections"]
                    if (d["conf"] > config["thresholds"]["megadetector_confidence"])
                    and (d["category"] == "1")
                ]
                if not md_detections_filtered:
                    species = "none"
                    output_dir = f"{image_training_output}/{config['output_folders']['discard_subfolder']}"
                    os.makedirs(output_dir, exist_ok=True)
                    output_filename = (
                        f"{output_dir}/{img_datetime}_{camera}_{species}.JPG"
                    )
                    cv2.imwrite(output_filename, image)
                    digivol_output.append(
                        {
                            "blank_path": image_path,
                            "annotated_path": output_filename,
                            "species": species,
                            "confidence": "none",
                            "bbox": "none",
                            "time": img_datetime,
                            "model": model_required,
                            "camera": camera,
                            "detection_conf": "none",
                        }
                    )
                    all_images.append(
                        {
                            "blank_path": image_path,
                            "annotated_path": output_filename,
                            "species": species,
                            "confidence": "none",
                            "bbox": "none",
                            "time": img_datetime,
                            "model": model_required,
                            "camera": camera,
                            "detection_conf": "none",
                            "cropped": "NO",
                            "cropped_path": "none",
                        }
                    )
                else:
                    for idx, detection in enumerate(md_detections_filtered):
                        bbox_normalized = detection["bbox"]
                        species = "animal"
                        output_dir_annotated = f"{image_training_output}/{config['output_folders']['further_classification_subfolder']}/{config['output_folders']['annotated_subfolder']}"
                        os.makedirs(output_dir_annotated, exist_ok=True)

                        output_dir = f"{image_training_output}/{config['output_folders']['further_classification_subfolder']}/{config['output_folders']['blank_subfolder']}"
                        os.makedirs(output_dir, exist_ok=True)

                        output_filename_annotated = f"{output_dir_annotated}/{img_datetime}_{camera}_{species}.JPG"
                        output_filename = (
                            f"{output_dir}/{img_datetime}_{camera}_{species}.JPG"
                        )
                        draw_bounding_boxes_on_file(
                            image_path,
                            output_filename_annotated,
                            md_detections_filtered,
                        )
                        cv2.imwrite(output_filename, image)
                        digivol_output.append(
                            {
                                "blank_path": image_path,
                                "annotated_path": output_filename_annotated,
                                "species": species,
                                "confidence": "none",
                                "bbox": bbox_normalized,
                                "time": img_datetime,
                                "model": model_required,
                                "camera": camera,
                                "detection_conf": "none",
                            }
                        )
                        all_images.append(
                            {
                                "blank_path": image_path,
                                "annotated_path": output_filename_annotated,
                                "species": species,
                                "confidence": "none",
                                "bbox": bbox_normalized,
                                "time": img_datetime,
                                "model": model_required,
                                "camera": camera,
                                "detection_conf": "none",
                                "cropped": "NO",
                                "cropped_path": "none",
                            }
                        )
            else:
                for idx, detection in enumerate(above):
                    bbox_normalized = detection.cpu().xyxy

                    cropped_img = crop_image(image, bbox_normalized)

                    bbox_normalized = bbox_normalized.tolist()
                    detection_conf = detection.conf.item()
                    print(detection_conf)
                    result = model.predict(
                        cropped_img,
                        save=False,  # I don't save anything here because I want to sort into folders
                        save_txt=False,
                        save_conf=False,
                    )[0]
                    species = result.names[result.probs.top1]
                    conf = result.cpu().probs.top1conf.item()
                    if conf < config["thresholds"]["classification_confidence_high"]:
                        output_dir = f"{image_training_output}/classification/{model_required}/low_confidence/{species}"
                    else:
                        output_dir = f"{image_training_output}/classification/{model_required}/{species}"

                    print(f"Saving to {output_dir}")
                    os.makedirs(output_dir, exist_ok=True)
                    output_filename = (
                        f"{output_dir}/{camera}_{img_datetime}_{species}_{idx}.JPG"
                    )
                    cv2.imwrite(output_filename, cropped_img)

                    all_detections.append(
                        {
                            "original_path": [image_path],
                            "cropped_path": [output_filename],
                            "species": [species],
                            "confidence": [conf],
                            "bbox": [bbox_normalized],
                            "time": [img_datetime],
                            "model": [model_required],
                            "camera": [camera],
                            "detection_conf": [detection_conf],
                        }
                    )
                    all_images.append(
                        {
                            "blank_path": image_path,
                            "annotated_path": "none",
                            "cropped_path": output_filename,
                            "species": species,
                            "confidence": conf,
                            "bbox": bbox_normalized,
                            "time": img_datetime,
                            "model": model_required,
                            "camera": camera,
                            "detection_conf": detection_conf,
                            "cropped": "YES",
                        }
                    )

        if all_detections:
            final_df = pd.DataFrame(all_detections)
            final_df.to_csv(
                f"{image_source_directory}/output_{camera}.csv", index=False
            )
        if digivol_output:
            # Concatenate all dictionaries into a single DataFrame
            digivol_df = pd.DataFrame(digivol_output)
            digivol_df.to_csv(
                f"{image_source_directory}/digivol_output_{camera}.csv", index=False
            )
    if all_images:
        final_df = pd.DataFrame(all_images)
        final_df.to_csv(
            f"{base_dir}/config/config_{datetime.today().strftime('%Y-%m-%d')}.csv",
            index=False,
        )
