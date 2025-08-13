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
        stdin, stdout, stderr = ssh_client.exec_command(command, get_pty=True)
        for line in iter(stdout.readline, ""):
            print(line, end="")
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


if __name__ == "__main__":
    # Ensure paramiko is installed: pip install paramiko
    # Ensure rclone is installed and configured on both machines
    # Ensure Python and R are in PATH on respective machines
    # Ensure all scripts are in their expected locations
    print("hello")