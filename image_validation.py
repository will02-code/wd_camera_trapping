import os
import time
import shutil
import pandas as pd
import cv2
from ultralytics import YOLO

validation_root_directory = "C:\\Users\\willo\\OneDrive - UNSW\\Documents\\Work\\CES\\Wild Deserts\\Image classification\\validation"
model_path = "C:\\Users\\willo\\OneDrive - UNSW\\Documents\\Work\\CES\\Wild Deserts\\Image classification\\training_output\\four_classes_070125\\final_output\\content\\runs\\detect\\train3\\weights\\best.pt"
directory = "C:\\Users\\willo\\OneDrive - UNSW\\Documents\\Work\\CES\\Wild Deserts\\Image classification\\validation\\all_images\\FC2"

def load_model(model_path):
    return YOLO(model_path)

def predict_images(model, directory, validation_root_directory):
    return model.predict(
        source=os.path.join(directory),
        save=False,
        save_txt=False,
        save_conf=False,
        imgsz=[1920, 1088],
        conf=0.1,
        iou=0.5,
        augment=True,
        project=f"{validation_root_directory}/output3/labels_images/predict",
        name="test"
    )

def process_predictions(predictions, model, validation_root_directory):
    all_detections = []
    for i in range(len(predictions)):
        if predictions[i].boxes:  # Check if there are any boxes in the prediction
            species = model.names[int(predictions[i].boxes.cls[0])]
            conf = round(float(predictions[i].boxes.conf[0]), 3)
            path = predictions[i].path
            name = os.path.splitext(os.path.basename(predictions[i].path))[0]
            bbox = predictions[i].boxes.xywh.tolist()
            newpath = f"{validation_root_directory}/output3/{species}"  # save the annotated image to the species folder
            os.makedirs(newpath, exist_ok=True)
            predictions[i].save(f"{newpath}/{name}.jpg")
            
            # save the original image to the images folder
            os.makedirs(f"{validation_root_directory}/output3/labels_images/predict/images", exist_ok=True)
            shutil.copy(path, f"{validation_root_directory}/output3/labels_images/predict/images")
            path_original = f"{validation_root_directory}/output3/labels_images/predict/images/{name}.JPG"
            
            # save the annotated image to the annotated folder
            path_annotated = f"{validation_root_directory}/output3/labels_images/predict/annotated/{name}.JPG"
            os.makedirs(f"{validation_root_directory}/output3/labels_images/predict/annotated", exist_ok=True)
            predictions[i].save(f"{validation_root_directory}/output3/labels_images/predict/annotated/{name}.JPG")
            
            # save the label file to the labels folder
            labels_path = f"{validation_root_directory}/output3/labels_images/predict/labels/{name}.txt"
            predictions[i].save_txt(labels_path)
            
            # Display the annotated image in a pop-up window with smaller size
            img = cv2.imread(path_annotated)
            img_resized = cv2.resize(img, (800, 600))  # Resize the image to 800x600
            window_name = f"Species: {species}, Confidence: {conf}"
            cv2.imshow(window_name, img_resized)
            cv2.moveWindow(window_name, 0, 0)  # Move the window to the top left corner
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Prompt user for input
            result = input("Is the detection correct? (y/n): ").strip().lower()
            
            # save the data to a dataframe
            df = pd.DataFrame({'path_original': path_original, 
                               'species': species, 
                               'confidence': conf, 
                               'bbox': bbox, 
                               "path_annotated": path_annotated,
                               "label_path": labels_path,
                               "results": result})
            
            # append the dataframe to the list of all detections
            all_detections.append(df)
        else:
            print(f"No detections for prediction {i}")
    return all_detections

def save_results(all_detections, validation_root_directory):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if all_detections:
        final_df = pd.concat(all_detections)
        final_df.to_csv(f"{validation_root_directory}/output3/detections_{timestamp}.csv", index=False)

def main():
    model = load_model(model_path)
    predictions = predict_images(model, directory, validation_root_directory)
    all_detections = process_predictions(predictions, model, validation_root_directory)
    save_results(all_detections, validation_root_directory)

if __name__ == "__main__":
    main()
