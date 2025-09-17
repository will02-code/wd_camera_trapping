#!/usr/bin/env python

import shutil
import pandas as pd
import regex as re
import yaml
import os
import cv2
import ast
import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import argparse

parser = argparse.ArgumentParser()


def crop_image(img, coords):
    try:
        # Simulate .int().item() behavior for regular Python floats
        x_min = int(coords[0][0])

        y_min = int(coords[0][1])
        x_max = int(coords[0][2])
        y_max = int(coords[0][3])

        cropped_img = img[y_min:y_max, x_min:x_max]
        return cropped_img

    except ValueError:
        print(
            f"Error: Invalid normalized coordinates format: {coords}. Expected 'x y w h'."
        )
        return None
    except IndexError:
        print(
            f"Error: Coords list is not in the expected format. Expected [[x, y, w, h]]. Got {coords}"
        )
        return None


def draw_bounding_box_on_image(
    image,
    coords,
    color="red",
    thickness=4,
    use_normalized_coordinates=True,
):
    """Adds a bounding box to an image.

    Bounding box coordinates can be specified in either absolute (pixel) or
    normalized coordinates by setting the use_normalized_coordinates argument.

    Each string in display_str_list is displayed on a separate line above the
    bounding box in black text on a rectangle filled with the input 'color'.
    If the top of the bounding box extends to the edge of the image, the strings
    are displayed below the bounding box.

    Args:
    image: a PIL.Image object.
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    use_normalized_coordinates: If True (default), treat coordinates
        ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
        coordinates as absolute.
    """
    if isinstance(coords[0], list):
        # Nested format: [[x, y, w, h]]
        bbox = coords[0]
    else:
        # Flat format: [x, y, w, h]
        bbox = coords
    xmin = int(bbox[0])  # x coordinate of top-left corner
    ymin = int(bbox[1])  # y coordinate of top-left corner
    xmax = int(bbox[2])  # x coordinate of bottom-right corner
    ymax = int(bbox[3])  # y coordinate of bottom-right corner

    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (
            xmin * im_width,
            xmax * im_width,
            ymin * im_height,
            ymax * im_height,
        )
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    if thickness > 0:
        draw.line(
            [(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
            width=thickness,
            fill=color,
        )

        return image


if __name__ == "__main__":
    parser.add_argument("-c", "--config_df", help="Config file")
    parser.add_argument("-d", "--target_dir", help="Target directory")
    args = parser.parse_args()
    # print(args.config_df)
    config_df = pd.read_csv(args.config_df)

    target_dir = args.target_dir

    unique_species = [
        x
        for x in config_df["species"].unique().tolist()
        if x != "none" and x != "animal" and x != "vehicle" and x != "person"
    ]

    remote_path = os.path.join(
        re.split(
            os.path.basename(target_dir),
            config_df["blank_path"].iloc[0],
        )[0],
        os.path.basename(target_dir),
    )
    print(remote_path)
    for line in config_df.iterrows():
        # print(line[0])
        blank_path = line[1]["blank_path"]

        new_blank_path = re.sub(remote_path, target_dir, blank_path)

        # print(new_blank_path)
        image = cv2.imread(new_blank_path)
        print(f"Processing image: {new_blank_path}")
        annotated_path = line[1]["annotated_path"]
        # print("annotated:", annotated_path)
        if annotated_path != "none":
            new_annotated_path = re.sub(remote_path, target_dir, annotated_path)
            annotated_dir = os.path.dirname(new_annotated_path)
            # print(annotated_dir)
            os.makedirs(annotated_dir, exist_ok=True)
            if line[1]["species"] == "none":
                cv2.imwrite(new_annotated_path, img=image)
            elif line[1]["species"] == "animal":
                new_annotated_path_blank = re.sub(
                    "annotated", "annotated/images", new_annotated_path
                )
                new_annotated_path_blank_dir = os.path.dirname(new_annotated_path_blank)
                os.makedirs(new_annotated_path_blank_dir, exist_ok=True)
                cv2.imwrite(new_annotated_path_blank, img=image)
                # Save the bounding box coordinates in a text file
                new_annotated_path_label = re.sub(
                    "images", "labels", new_annotated_path_blank
                )
                new_annotated_path_label = re.sub(
                    ".JPG", ".txt", new_annotated_path_label
                )
                new_annotated_path_label_dir = os.path.dirname(new_annotated_path_label)
                os.makedirs(new_annotated_path_label_dir, exist_ok=True)
                bbox = [
                    float(x.strip()) for x in line[1]["bbox"].strip("[]").split(",")
                ]
                xc, yc, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                x_new = xc + (w / 2)
                y_new = yc + (h / 2)
                correct_bbox_calculated = [x_new, y_new, w, h]

                # print("correct bbox:", correct_bbox_calculated)
                with open(new_annotated_path_label, "w") as f:
                    # Format the class ID and the calculated bounding box coordinates
                    # into the desired string format: 'class x_new y_new w h'
                    formatted_line = (
                        "0 "  # Class ID
                        f"{correct_bbox_calculated[0]:.4f} "  # x_new, formatted to 6 decimal places
                        f"{correct_bbox_calculated[1]:.4f} "  # y_new, formatted to 6 decimal places
                        f"{correct_bbox_calculated[2]:.4f} "  # w, formatted to 6 decimal places
                        f"{correct_bbox_calculated[3]:.4f}"  # h, formatted to 6 decimal places (no trailing space)
                    )
                    f.write(formatted_line)

        cropped_path = line[1]["cropped_path"]
        # print(cropped_path)

        if cropped_path != "none" and line[1]["species"] != "animal":
            new_cropped_path = re.sub(remote_path, target_dir, cropped_path)
            # print(new_cropped_path)
            coords_list = ast.literal_eval(line[1]["bbox"])

            cropped_img = crop_image(image, coords_list)
            # print(new_cropped_path)

            cropped_dir = os.path.dirname(new_cropped_path)
            os.makedirs(cropped_dir, exist_ok=True)

            cv2.imwrite(new_cropped_path, cropped_img)

            full_size_class_path = re.sub(
                "/classification/", "/full_size_classification/", new_cropped_path
            )
            full_size_class_dir = os.path.dirname(full_size_class_path)

            os.makedirs(full_size_class_dir, exist_ok=True)
            full_size_class_image = Image.fromarray(image)
            full_size_class_image = draw_bounding_box_on_image(
                full_size_class_image,
                coords_list,
                color="red",
                thickness=4,
                use_normalized_coordinates=False,
            )
            full_size_class_image.save(full_size_class_path)
