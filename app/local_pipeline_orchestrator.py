import subprocess
import paramiko  # or use os.system for simpler ssh, but paramiko is more robust
import datetime
import os
import yaml

CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_TEMPLATE_PATH = os.path.join(CURRENT_SCRIPT_DIR, "config.yaml")
try:
    with open(CONFIG_TEMPLATE_PATH, "r") as f:
        CONFIG = yaml.safe_load(f)
except FileNotFoundError:
    print("Error: config.yaml not found. Please create it based on the template.")
    exit(1)
except yaml.YAMLError as e:
    print(f"Error parsing config.yaml: {e}")
    exit(1)


# --- Configuration (read from a separate config.ini or similar) ---
LOCAL_ORIGINAL_IMAGES_BASE_PATH = CONFIG[
    "local"
][
    "base_images_path"
]  # This is the base path where original images are stored (e.g.the hard drive folder with ALLL the monthly folders)
LOCAL_PIPELINE_SCRIPTS_PATH = os.path.join(
    CURRENT_SCRIPT_DIR, CONFIG["local"]["scripts_folder"]
)  # This is the path to the local scripts folder where all scripts are stored. It is RELATIVE to where you are located in the terminal
LOCAL_CONFIG_PATH = os.path.join(
    CURRENT_SCRIPT_DIR, CONFIG["local"]["config_folder"]
)  # This is the path to the local config folder where all config files are stored. It is RELATIVE to where you are located in the terminal
REMOTE_HOST = CONFIG[
    "remote"
][
    "host"
]  # This is my username for remote access to the lab computer. Much much faster to do this than process everything locally
REMOTE_SCRIPTS_PATH = CONFIG[
    "remote"
][
    "scripts_base_path"
]  # Relative to remote user's home or specific path. the full path is actually home/willwright/Documents/AI_identification/, and we cd into Documents (we land in willwright)
REMOTE_OUTPUTS_PATH = CONFIG["remote"]["outputs_base_path"]  # As above
RCLONE_REMOTE_NAME = CONFIG[
    "cloud"
][
    "rclone_remote_name"
]  # This is the name of the rclone remote we have set up for OneDrive. It is set up in the rclone config file on the remote machine
BASE_CLOUD_PATH = CONFIG[
    "cloud"
][
    "base_cloud_path"
]  # This is the base path for the cloud storage (e.g., OneDrive, Google Drive, etc.). It is RELATIVE to the rclone remote name

ONEDRIVE_REMOTE_PATH = f"{RCLONE_REMOTE_NAME}:{BASE_CLOUD_PATH}"  # can change after we have set up the remote rclone config. This is the path to the OneDrive folder where we store all our data

ONEDRIVE_REMOTE_CONFIG_PATH = f"{ONEDRIVE_REMOTE_PATH}/{CONFIG['cloud']['config_cloud_path']}"  # This is the path to the OneDrive folder where we store all our config files. It is RELATIVE to the rclone remote name
ONEDRIVE_REMOTE_DATA_PATH = f"{ONEDRIVE_REMOTE_PATH}/{CONFIG['cloud']['raw_data_cloud_path']}"  # This is the path to the OneDrive folder where we store all our data. It is RELATIVE to the rclone remote name
ONEDRIVE_REMOTE_OUTPUT_PATH = f"{ONEDRIVE_REMOTE_PATH}/{CONFIG['cloud']['final_outputs_cloud_path']}"  # This is the path to the OneDrive folder where we store all our outputs. It is RELATIVE to the rclone remote name
RCLONE_PATH = CONFIG["local"][
    "rclone_path"
]  # Path to rclone executable on local machine. Will be different for everyone
RSCRIPT_PATH = CONFIG["local"][
    "rscript_path"
]  # Path to Rscript executable on local machine. Will be different for everyone
REMOTE_CONDA_INIT_SCRIPT = CONFIG[
    "remote"
][
    "conda_init_script"
]  # Path to conda init script on remote machine. This is where the conda environment is set up
FINAL_OUTPUTS_PATH = os.path.join(
    CURRENT_SCRIPT_DIR, CONFIG["local"]["final_outputs_path"]
)  # This is the path to the local folder where we store all our final outputs. It is RELATIVE to the scripts folder


def run_remote_command(command):
    print(f"\n--- Running Remote Command on {REMOTE_HOST}: {command} ---")
    # Using paramiko for robustness, but can be simpler os.system(f"ssh {REMOTE_HOST} '{command}'")
    try:
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(
            hostname=REMOTE_HOST.split("@")[1], username=REMOTE_HOST.split("@")[0]
        )
        stdin, stdout, stderr = ssh_client.exec_command(command)
        print(stdout.read().decode())
        error_output = stderr.read().decode()
        if error_output:
            print(f"--- Remote Error Output: ---\n{error_output}")
            # Decide if you want to exit on remote errors
            # exit(1)
        ssh_client.close()
        print("--- Remote Command Succeeded ---")
    except Exception as e:
        print(f"--- ERROR: Remote command failed: {e} ---")
        exit(1)


def get_latest_dated_file(directory, prefix, suffix):
    # More sophisticated logic might be needed for remote files via rclone ls
    # For local, os.listdir works
    files = [
        f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(suffix)
    ]
    if not files:
        return None
    # Assuming files are named like config_YYYY-MM-DD.csv
    files.sort(
        key=lambda x: datetime.datetime.strptime(x, f"{prefix}%Y-%m-%d{suffix}"),
        reverse=True,
    )
    return files[0]


def main_pipeline():
    # --- Step 1: Rename images on local computer ---
    local_image_folder_name = input(
        f"Enter the name of the original image folder (e.g., 2025_01_WCAM_originals) within {LOCAL_ORIGINAL_IMAGES_BASE_PATH}: "
    )
    full_local_image_path = os.path.join(
        LOCAL_ORIGINAL_IMAGES_BASE_PATH, local_image_folder_name
    )
    if not os.path.exists(full_local_image_path):
        print(f"Error: Folder '{full_local_image_path}' not found.")
        exit(1)
    output = subprocess.run(
        [
            "python",
            os.path.join(LOCAL_PIPELINE_SCRIPTS_PATH, "rename.py"),
            full_local_image_path,
        ],
        capture_output=True,
    )
    print(output.stdout.decode())

    # --- Step 2: Upload renamed images to Onedrive ---
    print(f"\n--- Uploading {local_image_folder_name} to Onedrive... ---")
    # Assuming rclone is set up correctly
    subprocess.run(
        [
            RCLONE_PATH,
            "sync",
            full_local_image_path,
            f"{ONEDRIVE_REMOTE_DATA_PATH}/{local_image_folder_name}",
            "--progress",
        ]
    )

    # --- Step 3 & 4 (Remote): Download, Classify, Upload Classification Config ---
    print("\n--- Triggering remote classification pipeline ---")
    # This remote script will handle downloading, renaming, classification, and uploading config_csv

    remote_classification_script = f"""

    source ~/miniconda3/bin/activate ai_training
    cd Documents
    rclone sync {ONEDRIVE_REMOTE_PATH} {REMOTE_OUTPUTS_PATH}
    python -c "import yaml; data = yaml.safe_load(open('{REMOTE_SCRIPTS_PATH}config.yaml')); data['paths']['target_dirs'] = ['{local_image_folder_name}']; yaml.safe_dump(data, open('{REMOTE_SCRIPTS_PATH}config.yaml', 'w'))"
    python {REMOTE_SCRIPTS_PATH}classification.py
    CLASSIF_CONFIG_FILE=$(ls -t wild_deserts_outputs/config/config_*.csv | head -1)
    if [ -z "$CLASSIF_CONFIG_FILE" ]; then
        echo "Error: Classification config file not found on remote machine."
        exit 1
    fi
    rclone copy "$CLASSIF_CONFIG_FILE" {ONEDRIVE_REMOTE_CONFIG_PATH}

    """
    run_remote_command(remote_classification_script)

    # --- Step 5 (Local): Download config and restructure files ---
    print("\n--- Downloading classification config from Onedrive ---")

    subprocess.run(
        [
            RCLONE_PATH,
            "copy",
            ONEDRIVE_REMOTE_CONFIG_PATH,
            LOCAL_CONFIG_PATH,
        ]
    )  # Download to local config folder

    # Find the most recently downloaded config file
    latest_config_file = get_latest_dated_file(LOCAL_CONFIG_PATH, "config_", ".csv")
    print(f"Config path: {LOCAL_CONFIG_PATH}")
    if not latest_config_file:
        print("Error: No classification config file found locally.")
        exit(1)

    full_ai_local_config_filepath = os.path.join(LOCAL_CONFIG_PATH, latest_config_file)
    print(f"Using classification config file: {full_ai_local_config_filepath}")

    output = subprocess.run(
        [
            "python",
            os.path.join(
                LOCAL_PIPELINE_SCRIPTS_PATH, "restructure_with_classifications.py"
            ),
            "-c",
            full_ai_local_config_filepath,
            "-d",
            LOCAL_ORIGINAL_IMAGES_BASE_PATH,
        ],
        capture_output=True,
    )
    print(output)
    # --- Step 6: Verify using digiKam software (Manual Step) ---
    input(
        "\n*** IMPORTANT: Please verify images using DigiKam software. Press Enter when done to continue... ***"
    )

    # --- Step 7 (Local): Get activity index and prepare training images ---
    print(
        "\n--- Running R script for activity index and training image preparation ---"
    )
    r_folders_arg = os.path.join(
        LOCAL_ORIGINAL_IMAGES_BASE_PATH, local_image_folder_name, "outputs"
    )  # Assuming this is where the outputs are for the R script to read

    r_temp_config = {
        "folders": r_folders_arg,
        "ai_classification_csv": full_ai_local_config_filepath,
        "parent_dir": LOCAL_ORIGINAL_IMAGES_BASE_PATH,
        "output_dir": FINAL_OUTPUTS_PATH,
        # Add other dynamic variables here if needed
        # "remote_dir": "/home/willwright/Documents/wild_deserts_outputs/", # If this also changes
        # "timezone": "Australia/Sydney",
    }
    temp_r_config_filepath = os.path.join(
        LOCAL_PIPELINE_SCRIPTS_PATH, "r_pipeline_temp_config.yaml"
    )
    try:
        with open(temp_r_config_filepath, "w") as f:
            yaml.dump(r_temp_config, f)
        print(f"Generated R config file: {temp_r_config_filepath}")
    except Exception as e:
        print(f"Error writing R config file: {e}")
        raise Exception("Failed to write R config.")

    outputs = subprocess.run(
        [
            RSCRIPT_PATH,
            os.path.join(LOCAL_PIPELINE_SCRIPTS_PATH, "generate_training_images.R"),
            temp_r_config_filepath,
        ],
        capture_output=True,
    )  # Pass the path to the config file
    latest_validated_config_file = get_latest_dated_file(
        FINAL_OUTPUTS_PATH,
        "validated_config",
        ".csv",
    )
    # Assuming R script outputs to the same directory
    if not latest_validated_config_file:
        print("Error: validated_config file not found after R script execution.")
        exit(1)
    full_local_validated_config_filepath = os.path.join(
        FINAL_OUTPUTS_PATH, latest_validated_config_file
    )
    print(f"Uploading validated config: {full_local_validated_config_filepath}")
    subprocess.run(
        [
            RCLONE_PATH,
            "copy",
            full_local_validated_config_filepath,
            ONEDRIVE_REMOTE_CONFIG_PATH,
        ],
        capture_output=True,
    )
    final_results_file = get_latest_dated_file(
        FINAL_OUTPUTS_PATH,
        "processed_camera_trap_data",
        ".csv",
    )
    final_results_file_full_path = os.path.join(FINAL_OUTPUTS_PATH, final_results_file)
    print(f"Uploading results: {final_results_file_full_path}")
    output = subprocess.run(
        [
            RCLONE_PATH,
            "copy",
            final_results_file_full_path,
            ONEDRIVE_REMOTE_OUTPUT_PATH,
        ],
        capture_output=True,
    )
    print(output)
    # --- Step 8 (Remote): Sort files into training dataset and upload to Onedrive ---
    print("\n--- Triggering remote training image generation and upload ---")
    # This remote script will handle downloading validated_config, sorting, and syncing to Onedrive
    # Need to correctly pass the validated_config filename to the remote script
    remote_training_script = f"""
    source ~/miniconda3/bin/activate ai_training
    cd Documents
    rclone sync {ONEDRIVE_REMOTE_CONFIG_PATH} ./{REMOTE_OUTPUTS_PATH}/config/ # Download latest validated config

    VALIDATED_CONFIG_FILE=$(ls -t ./{REMOTE_OUTPUTS_PATH}/config/validated_config*.csv | head -1)
    if [ -z "$VALIDATED_CONFIG_FILE" ]; then
        echo "Error: Validated config file not found on remote machine."
        exit 1
    fi
    echo "Using remote validated config: $VALIDATED_CONFIG_FILE"

    python {REMOTE_SCRIPTS_PATH}generate_training_images.py -vc "$VALIDATED_CONFIG_FILE"

    rclone copy ./{REMOTE_OUTPUTS_PATH}/ {ONEDRIVE_REMOTE_PATH}  # Syncs remote wild_deserts_outputs to Onedrive
    echo "Rclone sync completed."

    """

    run_remote_command(remote_training_script)


if __name__ == "__main__":
    # Ensure paramiko is installed: pip install paramiko
    # Ensure rclone is installed and configured on both machines
    # Ensure Python and R are in PATH on respective machines
    # Ensure all scripts are in their expected locations
    main_pipeline()
