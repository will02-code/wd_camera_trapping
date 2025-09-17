@echo off
ECHO "Starting the automated setup process..."

REM --- 1. Install Conda ---
ECHO "1/4: Checking for Conda installation..."
if exist "%USERPROFILE%\miniforge3" (
    ECHO "Conda is already installed. Skipping installation."
) else (
    ECHO "Conda not found. Downloading and installing Miniconda..."
    REM You should provide a URL to a stable Miniconda installer
    start /wait "" Miniforge3-Windows-x86_64.exe /InstallationType=JustMe /RegisterPython=0 /S /D=%UserProfile%\Miniforge3
    ECHO "Conda installation complete."
)

REM --- 2. Install R and R Packages ---
ECHO "2/4: Checking for R installation..."
if exist "C:\Program Files\R\R-4.4.2\bin\R.exe" (
    ECHO "R is already installed. Skipping installation."
) else (
    ECHO "R not found. Please install R from CRAN. This script will pause."
    ECHO "After installation, press any key to continue."
    pause
    ECHO "Proceeding with R package installation."
)

REM --- Install R packages using R.exe ---
ECHO "Installing R packages..."
"C:\Program Files\R\R-4.4.2\bin\R.exe" --vanilla -e "install.packages(c('tidyverse', 'exiftoolr', 'fs', 'glue', 'here', 'yaml', 'DBI', 'hms'), repos='https://cloud.r-project.org')"
ECHO "R packages installed."


REM --- 4. Install Rclone ---
ECHO "4/4: Downloading and installing Rclone..."
winget install Rclone.Rclone
ECHO "Rclone installed to C:\rclone."

ECHO "Setup complete! Please configure Rclone before running your scripts."
pause

