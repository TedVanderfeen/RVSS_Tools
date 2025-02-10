# RVSS Tools Scripts Documentation

Collection of Python scripts for things relating to the [RVSS_Need4Speed](https://github.com/rvss-australia/RVSS_Need4Speed) challenge.

## Scripts Overview

### vid_stitch.py
Creates videos from image sequences with angle overlays. 
- Processes sequences of images named in format `000001_angle.jpg`
- Adds angle information as text overlay
- Outputs MP4 video with configurable FPS and duration
- Usage: `python vid_stitch.py --input_folder /path/to/images --output video.mp4`

### dataset_clean.py 
Dataset visualization and cleaning tool.
- GUI interface for reviewing image datasets
- Shows angle information from filenames
- Allows deletion of unwanted images
- Useful for cleaning training datasets

### test_model.py
Testing script for trained PyTorch models.
- Tests model predictions against ground truth angles
- Processes test images named as `000033_0.20.jpg` format
- Configurable model path and test folder
- Usage: `python test_model.py --model_path model.pth --folder_path /test/images`

### bot_stop.py
Connects to the robot and sends a stop command if you get a crash when collecting data or deploying a model and the robot just keeps driving
- Usage: `python bot_stop.py --ip 192.168.1.xxx`

### get_image.py
Requests a single image from the camera onboard the PenguinPi and saves it as `image_x.jpg` depending on how many images you have grabbed before.
Useful for grabbing images of the stop sign for fine-tuning thresholding or some other method.
- Usage: `python get_image.py --ip 192.168.1.xxx`

### check_battery.py
Request a battery reading from the PenguinPi and prints the result as a calculation of the typical voltages for 18650 battery cells.
- Usage: `python check_battery.py --ip 192.168.1.xxx`

### dataset_csv.py
Creates a CSV file of images and labels for a given folder. Useful for some existing tools for analysing image datasets.

  
## Installation
If you've already setup your RVSS mamba/conda environment, you can simply clone this repo and then run these scripts from the RVSS_Tools directory.
```bash
git clone https://github.com/TedVanderfeen/RVSS_Tools.git
cd RVSS_Tools
