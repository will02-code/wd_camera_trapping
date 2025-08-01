# Improved Camera Trap Data Processing Pipeline
# ==============================================================================

# Load required libraries
suppressPackageStartupMessages({
  library(tidyverse)
  library(exiftoolr)
  library(tictoc)
  library(fs)
  library(glue)
  library(here)
  library(furrr)
  library(yaml)
})

# Configuration
# ==============================================================================
CONFIG <- list(
  # These will be overridden by command-line arguments or a config file
  folders = NULL, # Will be set by argument
  detection_training_dir = "wild_deserts_outputs/training/detection_training",
  classification_training_dir = "wild_deserts_outputs/training/classification_training",
  ai_classification_csv = NULL, # Will be set by argument
  parent_dir = NULL, # Will be set by argument
  remote_dir = "/home/willwright/Documents/wild_deserts_outputs/raw_data/", # This seems fixed, or could be an arg too
  image_width = 2048,
  image_height = 1440,
  timezone = "Australia/Sydney", # This might be constant, or an arg
  species_classes = c(
    "Kangaroo",
    "Cat",
    "Rabbit",
    "Dingo",
    "Fox",
    "Bilby",
    "Quoll",
    "Unidentifiable",
    "Bettong",
    "Crest-tailed mulgara",
    "Dusky hopping mouse",
    "Golden bandicoot",
    "Greater bilby",
    "Western quoll"
  ),
  species_regex = "Bilby|blobs|Cat|Dingo|Fox|Kangaroo|Quoll|non_target|Rabbit",
  threshold_for_timeblocks = 600, # Default threshold of 10 minutes (600 seconds)
  output_dir = NULL, # This will be set dynamically
  output_csv_name = paste0("processed_camera_trap_data", Sys.Date(), ".csv") # Use Sys.Date() for current date
)


# --- Argument Parsing for Config File Path ---
args <- commandArgs(trailingOnly = TRUE)
config_filepath <- NULL

if (length(args) > 0 && !is.na(args[1])) {
  config_filepath <- args[1]
  message(glue("Attempting to load config from: {config_filepath}"))
  if (file.exists(config_filepath)) {
    # Load YAML config
    external_config <- yaml::read_yaml(config_filepath)
    # Update default CONFIG with values from the external file
    CONFIG <- modifyList(CONFIG, external_config)
    message("External configuration loaded and applied.")
  } else {
    warning(glue(
      "Specified config file not found: {config_filepath}. Using default CONFIG."
    ))
  }
} else {
  message("No external config file specified. Using default CONFIG.")
}

# Ensure critical CONFIG values are present after loading/defaults
if (
  is.null(CONFIG$folders) ||
    is.null(CONFIG$ai_classification_csv) ||
    is.null(CONFIG$parent_dir)
) {
  stop(
    "Critical configuration variables (folders, ai_classification_csv, parent_dir) are not set. Ensure they are provided via config file or defined as defaults."
  )
}


# Ensure required configurations are set after argument parsing
if (
  is.null(CONFIG$folders) ||
    is.null(CONFIG$ai_classification_csv) ||
    is.null(CONFIG$parent_dir)
) {
  stop(
    "Critical configuration variables (folders, ai_classification_csv, parent_dir) are not set. Ensure they are provided via command-line arguments or defined as defaults."
  )
}


# Setup parallel processing
plan(multisession, workers = availableCores() - 1)

# Helper Functions
# ==============================================================================

#' Safely extract metadata from images
#' @param images Vector of image paths
#' @param tags EXIF tags to extract
extract_image_metadata <- function(images, tags = "Categories") {
  cat(glue("Processing {length(images)} images...\n"))

  tic()
  tryCatch(
    {
      metadata <- exiftoolr::exif_read(images, tags = tags, quiet = FALSE)
      toc()
      return(metadata)
    },
    error = function(e) {
      stop(glue("Failed to extract metadata: {e$message}"))
    }
  )
}

#' Clean and process image metadata
#' @param data_list Raw metadata from exiftoolr
process_metadata <- function(data_list) {
  data_list %>%
    mutate(
      Categories = str_remove_all(Categories, "Categor"),
      correct_incorrect = str_extract(Categories, "correct|incorrect"),
      predicted_species = str_extract(SourceFile, CONFIG$species_regex),
      correct_species = case_when(
        correct_incorrect == "incorrect" ~
          str_extract(Categories, CONFIG$species_regex),
        correct_incorrect == "correct" & !str_detect(SourceFile, "discard") ~
          predicted_species,
        str_detect(SourceFile, "further_verification") ~
          str_extract(Categories, CONFIG$species_regex),
        str_detect(SourceFile, "discard") ~ "empty",
        .default = NA_character_
      ),
      discard = str_detect(SourceFile, "discard"),
      confidence = if_else(
        str_detect(SourceFile, "low_confidence"),
        "low_confidence",
        "high_confidence"
      ),
      camera = str_extract(SourceFile, "[WP]CAM\\d{2}"),
      datetime = str_extract(
        SourceFile,
        "\\d{4}_\\d{2}_\\d{2}_\\d{2}_\\d{2}_\\d{2}"
      ) %>%
        ymd_hms(tz = CONFIG$timezone),
      id = glue("{camera}_{datetime}")
    ) %>%
    mutate(
      correct_species = factor(correct_species),
      predicted_species = factor(predicted_species)
    )
}

#' Convert bounding box coordinates from xyxy to YOLO format
#' @param bbox_string String representation of bounding box coordinates
#' @param image_width Image width in pixels
#' @param image_height Image height in pixels
#' @param round_digits Number of decimal places to round to
convert_xyxy_to_yolo <- function(
  bbox_string,
  image_width = CONFIG$image_width,
  image_height = CONFIG$image_height,
  round_digits = 4
) {
  # Input validation
  if (is.na(bbox_string) || bbox_string == "") {
    return(NA_character_)
  }

  tryCatch(
    {
      coords <- bbox_string %>%
        str_remove_all("[\\[\\]]") %>%
        str_split(", ") %>%
        pluck(1) %>%
        as.numeric()

      if (length(coords) != 4) {
        return(NA_character_)
      }

      x_min <- coords[1]
      y_min <- coords[2]
      x_max <- coords[3]
      y_max <- coords[4]

      # Calculate YOLO format coordinates
      box_width <- x_max - x_min
      box_height <- y_max - y_min
      x_center_norm <- (x_min + box_width / 2) / image_width
      y_center_norm <- (y_min + box_height / 2) / image_height
      width_norm <- box_width / image_width
      height_norm <- box_height / image_height

      paste(
        round(x_center_norm, round_digits),
        round(y_center_norm, round_digits),
        round(width_norm, round_digits),
        round(height_norm, round_digits)
      )
    },
    error = function(e) {
      warning(glue("Failed to convert bbox: {bbox_string} - {e$message}"))
      return(NA_character_)
    }
  )
}

#' Scale MegaDetector center coordinates
#' @param bbox_string String representation of bounding box coordinates
#' @param width Image width
#' @param height Image height
#' @param round_digits Number of decimal places to round to
scale_md_centre <- function(
  bbox_string,
  width = CONFIG$image_width,
  height = CONFIG$image_height,
  round_digits = 3
) {
  if (is.na(bbox_string) || bbox_string == "") {
    return(NA_character_)
  }

  tryCatch(
    {
      coords <- bbox_string %>%
        str_remove_all("[\\[\\]]") %>%
        str_split(", ") %>%
        pluck(1) %>%
        as.numeric()

      if (length(coords) != 4) {
        return(NA_character_)
      }

      x_min <- coords[1]
      y_min <- coords[2]
      w <- coords[3]
      h <- coords[4]

      x_c <- x_min + w / 2
      y_c <- y_min + h / 2

      paste(
        round(x_c, round_digits),
        round(y_c, round_digits),
        round(w, round_digits),
        round(h, round_digits)
      )
    },
    error = function(e) {
      warning(glue("Failed to scale bbox: {bbox_string} - {e$message}"))
      return(NA_character_)
    }
  )
}

#' Safely copy files with error handling
#' @param df Dataframe with source and destination paths
#' @param source_col Column name for source paths
#' @param dest_col Column name for destination paths
#' @param file_type Type of file being copied (for logging)
safe_file_copy <- function(df, source_col, dest_col, file_type = "file") {
  cat(glue("Copying {nrow(df)} {file_type}s...\n"))

  success_count <- 0
  error_count <- 0

  for (i in seq_len(nrow(df))) {
    source <- df[[source_col]][i]
    dest <- df[[dest_col]][i]

    # Create destination directory
    dir_create(path_dir(dest), recurse = TRUE)

    result <- tryCatch(
      {
        if (file_exists(source)) {
          file_copy(source, dest, overwrite = TRUE)
          success_count <<- success_count + 1
          if (i %% 50 == 0 || i == nrow(df)) {
            print(paste(i, "of", nrow(df), "files copied"))
          }
          TRUE
        } else {
          cat(glue("⚠ Source {file_type} not found: {source}\n"))
          error_count <<- error_count + 1
          FALSE
        }
      },
      error = function(e) {
        cat(glue(
          "✗ Failed to copy {file_type} {path_file(source)}: {e$message}\n"
        ))
        error_count <<- error_count + 1
        FALSE
      }
    )
  }

  cat(glue(
    "✓ {file_type} copy complete: {success_count} successful, {error_count} errors\n"
  ))
  return(list(success = success_count, errors = error_count))
}

#' Write YOLO labels to files
#' @param df Dataframe with labels and destination paths
write_yolo_labels <- function(df) {
  cat(glue("Writing {nrow(df)} YOLO label files...\n"))

  success_count <- 0
  error_count <- 0

  for (i in seq_len(nrow(df))) {
    label <- df$label[i]
    dest_path <- df$destination_labels[i]

    result <- tryCatch(
      {
        dir_create(path_dir(dest_path), recurse = TRUE)
        write_lines(label, dest_path)
        success_count <<- success_count + 1
        if (i %% 50 == 0 || i == nrow(df)) {
          print(paste(i, "of", nrow(df), "labels written"))
        }
        TRUE
      },
      error = function(e) {
        cat(glue(
          "✗ Failed to create label {path_file(dest_path)}: {e$message}\n"
        ))
        error_count <<- error_count + 1
        FALSE
      }
    )
  }

  cat(glue(
    "✓ Label writing complete: {success_count} successful, {error_count} errors\n"
  ))
  return(list(success = success_count, errors = error_count))
}

#' Get activity statitsics for a given dataframe. Generally after verification, though there is scope to
#' only use AI data. WIP.
#' @param df Dataframe with camera trap data. Needs headers SourceFile, correct_species, camera, datetime.

get_activity_statistics <- function(df) {
  group_id <- 1
  cleaned <- df |>
    select(SourceFile, correct_species, camera, datetime) |>
    filter(correct_species != "non_target", correct_species != "empty") |>
    arrange(datetime) |>
    mutate(time_block = NA_integer_)
  start_time <- cleaned$datetime[1]
  cleaned$time_block[1] <- group_id
  cameras <- unique(cleaned$camera)
  cleaned_by_camera <- tibble()
  # I loop through each camera; you may not need to do this depending on how your data is structured
  for (cam in cameras) {
    # print(cam)
    # Filter the data for the current camera and sort by datetime
    temp <- cleaned %>%
      arrange(camera) %>%
      filter(camera == cam) %>%
      arrange(datetime)
    # Loop through each row in the filtered data. All I do in this loop is assign a time_block to each row; one every 10 minutes for each camera. Look at the resulting df for this to make sense. But it's basically so I can process by 10 minute windows
    for (i in 2:nrow(temp)) {
      # set the threshold

      # If current datetime is more than 10 minutes after the start_time,
      # increment the group counter and update the start_time
      if (
        difftime(temp$datetime[i], start_time, units = "secs") >
          CONFIG$threshold
      ) {
        group_id <- group_id + 1
        start_time <- temp$datetime[i]
      }

      # Assign the current group_id
      temp$time_block[i] <- group_id
    }
    cleaned_by_camera <- bind_rows(cleaned_by_camera, temp)
  }

  cleaned_by_camera |>
    group_by(camera, datetime, correct_species) |>
    mutate(n = n()) |>
    arrange(camera, datetime) |>
    group_by(camera, time_block, correct_species) |>
    summarise(count = max(n), datetime = first(datetime)) |>
    write.csv(paste0(CONFIG$output_dir, "/", CONFIG$output_csv_name))
}

# Main Processing Functions
# ==============================================================================

#' Main function to process camera trap images
process_camera_trap_data <- function() {
  # 1. Load and process image metadata
  cat("Step 1: Loading image metadata...\n")
  images <- list.files(
    CONFIG$folders,
    full.names = TRUE,
    recursive = TRUE,
    pattern = "\\.(jpg|JPG)$"
  )
  # print(images)
  if (length(images) == 0) {
    stop("No images found in specified folders")
  }

  cat(glue("Found {length(images)} images\n"))

  # Extract metadata
  exifr::configure_exiftool()
  raw_metadata <- extract_image_metadata(images, tags = "Categories")

  # Process metadata and output activity data
  final_df <- process_metadata(raw_metadata)

  # 2. Process detection training data
  cat("\nStep 2: Processing detection training data...\n")
  detection <- process_detection_training(final_df)

  # 3. Process classification training data
  cat("\nStep 3: Processing classification training data...\n")
  classification <- process_classification_training(final_df)

  cat("\n✅ Pipeline completed successfully!\n")
  return(list(full_join(detection, classification, by = "id"), final_df))
}

#' Process detection training data
#' @param final_df Processed metadata dataframe
process_detection_training <- function(final_df) {
  # Process incorrect bounding boxes
  cat("Processing incorrect bounding boxes...\n")

  # Process all labeled detections
  cat("Processing all labeled detections...\n")

  # Load AI classification data safely
  if (!file_exists(CONFIG$ai_classification_csv)) {
    warning(glue(
      "AI classification CSV not found: {CONFIG$ai_classification_csv}"
    ))
    return(invisible(NULL))
  }

  ai_classification_df <- read_csv(
    CONFIG$ai_classification_csv,
    show_col_types = FALSE
  ) %>%
    mutate(
      filename = basename(blank_path),
      datetime = ymd_hms(time, tz = CONFIG$timezone),
      id = glue("{camera}_{datetime}")
    )

  incorrect_bounding <- final_df %>%
    filter(
      correct_incorrect == "incorrect",
      correct_species == "non_target"
    ) %>%
    left_join(ai_classification_df, by = "id") %>%

    mutate(
      detection_image_path = blank_path |> str_remove("will_drive/"),
      detection_label_path = str_replace(
        detection_image_path,
        "images",
        "labels"
      ) %>%
        str_replace("\\.jpg$", ".txt"),
      night_day = case_when(
        str_detect(SourceFile, "night") ~ "night",
        str_detect(SourceFile, "day") ~ "day",
        model == "day" ~ "day",
        model == "night" ~ "night",
        .default = "day"
      ),
      destination_images = glue(
        "{CONFIG$detection_training_dir}/{night_day}/images/{basename(detection_image_path)}"
      ),
      destination_labels = str_replace(
        destination_images,
        "images",
        "labels"
      ) %>%
        str_replace("\\.JPG$", ".txt"),
      label = NA_character_
    ) |>
    select(
      id,
      detection_image_path,
      destination_images,
      destination_labels,
      label
    )

  # if (nrow(incorrect_bounding) > 0) {
  #   results <- safe_file_copy(
  #     incorrect_bounding,
  #     "detection_image_path",
  #     "destination_images",
  #     "detection image"
  #   )
  #   cat(glue(
  #     "Processed {results$success} incorrect bounding images ({results$errors} errors)\n"
  #   ))
  # }

  all_labelled_detections <- final_df %>%
    left_join(ai_classification_df, by = "id") %>%
    filter(correct_species != "non_target") %>%
    mutate(
      detection_image_path = blank_path |> str_remove("will_drive/"),
      night_day = case_when(
        str_detect(SourceFile, "night") ~ "night",
        str_detect(SourceFile, "day") ~ "day",
        model == "day" ~ "day",
        model == "night" ~ "night",
        .default = "day"
      ),
      destination_images = glue(
        "{CONFIG$detection_training_dir}/{night_day}/images/{basename(detection_image_path)}"
      ),
      destination_labels = str_replace(
        destination_images,
        "images",
        "labels"
      ) %>%
        str_replace("\\.JPG$", ".txt"),
      label = case_when(
        correct_species == "empty" ~ NA_character_,
        species == "animal" ~
          glue(
            "{match(correct_species, CONFIG$species_classes) - 1} {map_chr(bbox, scale_md_centre)}"
          ),
        species != "animal" ~
          glue(
            "{match(correct_species, CONFIG$species_classes) - 1} {map_chr(bbox, convert_xyxy_to_yolo)}"
          ),
        .default = NA_character_
      ) %>%
        str_remove_all(",") %>%
        str_squish()
    ) %>%
    filter(!is.na(label), !is.na(detection_image_path)) %>%
    group_by(id) %>%
    distinct() %>%
    summarise(
      detection_image_path = first(detection_image_path),
      destination_images = first(destination_images),
      destination_labels = first(destination_labels),
      label = str_c(unique(label), collapse = "\n"),
      .groups = "drop"
    ) |>
    select(
      id,
      detection_image_path,
      destination_images,
      destination_labels,
      label
    ) |>
    bind_rows(incorrect_bounding)

  return(all_labelled_detections)
  # if (nrow(all_labelled_detections) > 0) {
  #   # Write labels
  #   label_results <- write_yolo_labels(all_labelled_detections)

  #   # Copy images
  #   image_results <- safe_file_copy(
  #     all_labelled_detections,
  #     "detection_image_path",
  #     "destination_images",
  #     "detection image"
  #   )

  #   cat(glue(
  #     "Processed {nrow(all_labelled_detections)} labeled detections: {image_results$success} images, {label_results$success} labels\n"
  #   ))
  # }
}

#' Process classification training data
#' @param final_df Processed metadata dataframe
process_classification_training <- function(final_df) {
  classification_data <- final_df %>%
    filter(correct_species != "non_target") %>%
    mutate(
      cropped_image = str_replace(
        SourceFile,
        CONFIG$parent_dir,
        CONFIG$remote_dir
      ),
      night_day = case_when(
        str_detect(cropped_image, "night") ~ "night",
        str_detect(cropped_image, "day") ~ "day",
        .default = NA_character_
      ),
      classification_dest_image = glue(
        "{CONFIG$classification_training_dir}/{night_day}/{correct_species}/{correct_incorrect}_{basename(cropped_image)}"
      )
    ) %>%
    filter(
      !is.na(cropped_image),
      !is.na(classification_dest_image),
      !is.na(night_day)
    ) |>
    select(
      cropped_image,
      classification_dest_image,
      id
    )
  return(classification_data)
  # if (nrow(classification_data) > 0) {
  #   results <- safe_file_copy(
  #     classification_data,
  #     "cropped_image",
  #     "classification_dest_image",
  #     "classification image"
  #   )

  #   cat(glue(
  #     "Processed {results$success} classification images ({results$errors} errors)\n"
  #   ))
  # }
}

# Execute Pipeline
# ==============================================================================

# Run the main processing function
cat("Executing R pipeline...\n")
final_results <- process_camera_trap_data()
file_paths <- final_results[[1]]
print(getwd())
validated_config_filename <- paste0(
  CONFIG$output_dir,
  "/validated_config",
  Sys.Date(),
  ".csv"
)
write.csv(
  file_paths,
  validated_config_filename, # Use the dynamically generated name
  row.names = FALSE
)
cat(glue("Validated config written to: {validated_config_filename}\n"))
get_activity_statistics(final_results[[2]])
cat(glue("Activity statistics written to: {CONFIG$output_csv_name}\n"))

cat("R script finished.\n")
