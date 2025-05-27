#
#
#
#
#
#
#
library(exiftoolr)
library(tidyverse)

# install exiftool
install_exiftool()
#
#
#

print("hello")



image_path = "E:/Wild deserts photos/2024_04_WCAM_originals/training/day/Kangaroo/WCAM03_2024_04_09_08_49_25_Kangaroo_0.jpg"



correct_incorrect = str_extract(exif$Categories, "correct|incorrect")
if (correct_incorrect == "incorrect"){
    correct_species = str_extract(exif$Categories, "Bilby|blobs|Cat|Dingo|Fox|Kangaroo|Quoll|non_target|Rabbit")
}else{
    correct_species = str_extract(image_path, "Bilby|blobs|Cat|Dingo|Fox|Kangaroo|Quoll|non_target|Rabbit")
}

images = list.files("E:/Wild deserts photos/2024_04_WCAM_originals/training/day", full.names =TRUE, recursive = TRUE)

data_list <- list()
for (image_path in images){
    exif <- exif_read(image_path) %>% select(Categories)
    correct_incorrect = str_extract(exif$Categories, "correct|incorrect")
    predicted_species = str_extract(image_path, "Bilby|blobs|Cat|Dingo|Fox|Kangaroo|Quoll|non_target|Rabbit")
    if (correct_incorrect == "incorrect"){
        correct_species = str_extract(exif$Categories, "Bilby|blobs|Cat|Dingo|Fox|Kangaroo|Quoll|non_target|Rabbit")
    }else{ # if correct
        correct_species = predicted_species
    }
    temp_df <- data.frame(
        image_path = image_path,
        predicted_species = predicted_species,
        correct_species = correct_species
    )
    data_list[[length(data_list) + 1]] <- temp_df

    # Print progress every 100 images
    if (length(data_list) %% 100 == 0) {
        print(length(data_list))
    }
}
final_df <- bind_rows(data_list)

#
#
#
#
