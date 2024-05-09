#!/usr/bin/python3
import time
from picamera2 import Picamera2, Preview
from libcamera import controls

import time
import datetime
from datetime import datetime

computerName = "Glowcaker_01"
import cv2

import csv
import numpy as np


import io
from PIL import Image
import piexif

print("---------------GLOWPI Scan--------------------")

print(" Time Stuff (cuz why not)")
now = datetime.now()
formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")  # Adjust the format as needed

print(f"Current time: {formatted_time}")

import os, platform
if platform.system() == "Windows":
	print(platform.uname().node)
else:
	computerName = os.uname()[1]
	print(os.uname()[1])   # doesnt work on windows



# ~~~~~~~~~~~~~~~~CAMERA STUFF ~~~~~~~~~~~~~~~~~~~~~~
def get_control_values(filepath):
    """Reads key-value pairs from the control file."""
    control_values = {}
    with open(filepath, "r") as file:
        for line in file:
            key, value = line.strip().split("=")
            control_values[key] = value
    return control_values

def load_camera_settings():
    """
    Reads camera settings from a CSV file and converts them to appropriate data types.

    Args:
        filepath (str): Path to the CSV file containing camera settings.

    Returns:
        dict: Dictionary containing camera settings with converted data types.

    Raises:
        ValueError: If an invalid value is encountered in the CSV file.
    """
    
    
    #first look for any updated CSV files on external media, we will prioritize those
    external_media_paths = ("/media", "/mnt")  # Common external media mount points
    default_path = "/home/pi/Desktop/Mothbox/scripts/StructuredLight/scan_camera_settings.csv"
    file_path=default_path

    found = 0
    for path in external_media_paths:
        if(found==0):
            files=os.listdir(path) #don't look for files recursively, only if new settings in top level
            if "camera_settings.csv" in files:
                file_path = os.path.join(root, "camera_settings.csv")
                print(f"Found settings on external media: {file_path}")
                found=1
                break
            else:
                print("No external settings here...")
                file_path=default_path

    if(found==0):
      
        #redundant but being extra safe
        print("No external settings, using internal csv")
        file_path=default_path


    try:
        with open(file_path) as csv_file:
            reader = csv.DictReader(csv_file)
            camera_settings = {}
            for row in reader:
                setting, value, details = row["SETTING"], row["VALUE"], row["DETAILS"]

                # Convert data types based on setting name (adjust as needed)
                if setting == "LensPosition":
                    try:
                        value = float(value)
                    except ValueError:
                        raise ValueError(f"Invalid value for LensPosition: {value}")
                elif setting == "AnalogueGain":
                    try:
                        value = float(value)
                    except ValueError:
                        raise ValueError(f"Invalid value for AnalogueGain: {value}")
                elif setting == "AeEnable" or setting == "AwbEnable":
                    value = value.lower() == "true"  # Convert to bool (adjust logic if needed)
                elif setting == "AwbMode" or setting == "AfTrigger" or setting == "AfRange"  or setting == "AfSpeed" or setting == "AfMode":
                    value=int(value)
                    #value = getattr(controls.AwbModeEnum, value)  # Access enum value
                    # Assuming AwbMode is a string representing an enum value
                    #pass  # No conversion needed for string
                elif setting == "ExposureTime":
                    try:
                        value = int(value)
                        middleexposure = value
                        print("middleexposurevalue ", middleexposure)
                    except ValueError:
                        raise ValueError(f"Invalid value for ExposureTime: {value}")
                else:
                    print(f"Warning: Unknown setting: {setting}. Ignoring.")

                camera_settings[setting] = value

        return camera_settings

    except FileNotFoundError as e:
        print(f"Error: CSV file not found: {file_path}")
        return None


def gray_andTakePhoto():
    # LensPosition: Manual focus, Set the lens position.
    now = datetime.now()
    timestamp = now.strftime("%Y_%m_%d__%H_%M_%S")  # Adjust the format as needed
    #timestamp = now.strftime("%y%m%d%H%M%S")
    

    ''''''
    if camera_settings:
        picam2.set_controls(camera_settings)
        picam2b.set_controls(camera_settings)

    else:
        print("can't set controls")
    ''''''

    time.sleep(1)
    picam2.start()
    picam2b.start()

    time.sleep(3)
    picam2.stop()
    picam2b.stop()



    exposureset_delay=.3 #values less than 5 don't seem to work! (unless you restart the cam!)
    requests = []  # Create an empty list to store requests
    requestsb = []  # Create an empty list to store requests

    PILs = []
    metadatas = []
    PILsb = []
    metadatasb = []
    
 
    i=0
    # Photo loop
         # Display each graycode pattern fullscreen
    cv2.namedWindow('Graycode Pattern', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Graycode Pattern', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # font 
    font = cv2.FONT_HERSHEY_SIMPLEX 
      
    # org 
    org = (50, 50) 
      
    # fontScale 
    fontScale = 2
       
    # Blue color in BGR 
    color = (255, 0, 0) 
      
    # Line thickness of 2 px 
    thickness = 3
       
    # Using cv2.putText() method 
    #image = cv2.putText(graycode_patterns[1][0], 'pre-show', org, font,fontScale, color, thickness, cv2.LINE_AA) 


    cv2.imshow('Graycode Pattern', whiteimg)

    #cv2.imshow('Graycode Pattern', graycode_patterns[1][0])
    picam2.start() #need to restart camera or wait a couple frames for settings to change
    picam2b.start() #need to restart camera or wait a couple frames for settings to change
    cv2.waitKey(1000)

    
    i=0
    for pattern in graycode_patterns[1]:
        start = time.time()
        
        #label scans for debugging
        #image = cv2.putText(pattern, str(i), org, font,fontScale, color, thickness, cv2.LINE_AA) 
        
        #display the pattern               
        cv2.imshow('Graycode Pattern', pattern)
        cv2.waitKey(10)


        
        request = picam2.capture_request(flush=True)
        requestb = picam2b.capture_request(flush=True)



        pilImage = request.make_image("main")
        pilImageb = requestb.make_image("main")

        PILs.append(pilImage)
        PILsb.append(pilImageb)


        request.release()
        requestb.release()


        #time.sleep(1)

        flashtime=time.time()-start
        print(str(i)+" picture take time: "+str(flashtime))
        i=i+1
        

    #capture the white image
    #display the pattern               
    cv2.imshow('Graycode Pattern', whiteimg)
    cv2.waitKey(10)
    
    request = picam2.capture_request(flush=True)
    requestb = picam2b.capture_request(flush=True)

    whitePIL = request.make_image("main")
    whitePILb = requestb.make_image("main")

    request.release()
    requestb.release()
    
    #capture the black image
    #display the pattern               
    cv2.imshow('Graycode Pattern', blackimg)
    cv2.waitKey(10)
    
    request = picam2.capture_request(flush=True)
    requestb = picam2b.capture_request(flush=True)

    blackPIL = request.make_image("main")
    blackPILb = requestb.make_image("main")

    request.release()
    requestb.release()



    picam2.stop()
    picam2b.stop()     
    
    #~~~ SAVE ALL THE IMAGES TO DISK ~~~~    
    
    base_path = "/home/pi/Desktop/Mothbox/scripts/StructuredLight/"

    # Create the folder structure
    folder_path = create_scan_folder(base_path)

    if folder_path:
      print(f"Created folder structure: {folder_path}")
    else:
      print("Failed to create folder structure.")
        
    # Saving loop (Cam 0 - A)
    cv2.destroyAllWindows()
    i=0
    for img in PILs:
          pil_image = img
          # Save the image using PIL to get the image data on disk
          folderPath= folder_path+"/a/" #can't use relative directories with cron
          #filepath = folderPath+"CamA_"+str(i)+".png" # If the extension is set to PNG, it will save a high quality PNG, BUTTT it takes like 4 seconds to save a single file.
          filepath = folderPath+"CamA_"+str(i)+".jpg" 
          img.save(filepath, quality=95)
          print("Image saved to "+filepath)
          i=i+1

    # Saving loop (Cam 1 - b)
    i=-1
    for img in PILsb:
          pil_image = img
          # Save the image using PIL to get the image data on disk
          folderPath= folder_path+"/b/" #can't use relative directories with cron
          #filepath = folderPath+"CamB_"+str(i)+".png" # If the extension is set to PNG, it will save a high quality PNG, BUTTT it takes like 4 seconds to save a single file.
          filepath = folderPath+"CamB_"+str(i)+".jpg" 
          img.save(filepath, quality=95)
          print("Image saved to "+filepath)
          i=i+1
      
    #save white and black images A
    folderPath= folder_path+"/a/" #can't use relative directories with cron
    #filepath = folderPath+"CamA_"+str(i)+".png" # If the extension is set to PNG, it will save a high quality PNG, BUTTT it takes like 4 seconds to save a single file.
    filepath = folderPath+"CamA_"+"WB_0"+".jpg" 
    whitePIL.save(filepath, quality=95)
    print("Image saved to "+filepath)
    
    filepath = folderPath+"CamA_"+"WB_1"+".jpg" 
    blackPIL.save(filepath, quality=95)
    print("Image saved to "+filepath)
    
    #save white and black images B
    folderPath= folder_path+"/b/" #can't use relative directories with cron
    #filepath = folderPath+"CamA_"+str(i)+".png" # If the extension is set to PNG, it will save a high quality PNG, BUTTT it takes like 4 seconds to save a single file.
    filepath = folderPath+"CamB_"+"WB_0"+".jpg" 
    whitePILb.save(filepath, quality=95)
    print("Image saved to "+filepath)
    
    filepath = folderPath+"CamB_"+"WB_1"+".jpg" 
    blackPILb.save(filepath, quality=95)
    print("Image saved to "+filepath)
      
      
      
      
      

def create_scan_folder(base_path):
  """Creates a folder with a timestamp and subfolders inside it.

  Args:
    base_path: The base path where the timestamped folder will be created.

  Returns:
    The path to the created folder or None if creation failed.
  """
  # Get current timestamp in a specific format
  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

  # Create the main folder with timestamp
  folder_name = f"Scan_{timestamp}"
  folder_path = os.path.join(base_path, folder_name)
  try:
    os.makedirs(folder_path)
  except OSError as e:
    print(f"Error creating folder: {e}")
    return None

  # Create subfolders within the main folder
  subfolders = ["a", "b", "decoded"]
  for subfolder in subfolders:
    subfolder_path = os.path.join(folder_path, subfolder)
    try:
      os.makedirs(subfolder_path)
    except OSError as e:
      print(f"Error creating subfolder {subfolder}: {e}")
      # Consider handling subfolder creation errors here (e.g., continue)

  return folder_path



# ~~~~~~~~~~~~ GRAYCODE STUFF ~~~~~~~~~~~

def generate_graycode(resolution):
  """Generates a sequence of Graycode patterns based on resolution.

  Args:
    resolution: A tuple representing the projector resolution (width, height).

  Returns:
    A list of NumPy arrays representing the Graycode patterns.
  """
  # Create a GrayCodePattern object with the provided resolution
  pattern = cv2.structured_light.GrayCodePattern.create(width=resolution[0], height=resolution[1])
  
 
 

  # Generate the graycode patterns
  pattern = pattern.generate()
  print(len(pattern[1]))
  # Create empty NumPy arrays for black and white images (adjust dtype if needed)
  blackImage = np.zeros_like(pattern[1][0], dtype=np.uint8)  # Same size and type as first pattern
  whiteImage = np.ones_like(pattern[1][0], dtype=np.uint8)*255  # Same size and type as first pattern
  
  #pattern.getImagesForShadowMasks(blackImage, whiteImage)


  return pattern, whiteImage, blackImage

def display_fullscreen(pattern):
  """Displays a NumPy array image fullscreen.

  Args:
    pattern: A NumPy array representing the image to display.
  """
  cv2.namedWindow('Graycode Pattern', cv2.WND_PROP_FULLSCREEN)
  cv2.setWindowProperty('Graycode Pattern', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
  cv2.imshow('Graycode Pattern', pattern)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

if __name__ == '__main__':
  # Define default projector resolution
  projector_divide_res = 12
  default_projector_res = (int(1920/projector_divide_res), int(1080/projector_divide_res))
  default_camA_res = (1920, 1080)
  default_camB_res = (1920, 1080)

  # # Get projector resolution from user input (or use default)
  # projector_res_str = input("Enter projector resolution (width height) [default: {}]: ".format(default_projector_res))
  # if projector_res_str:
    # projector_res = tuple(map(int, projector_res_str.split()))
  # else:
    # projector_res = default_projector_res

  projector_res = default_projector_res


  control_values = get_control_values("/home/pi/Desktop/Mothbox/controls.txt")


  picam2 = Picamera2(0)
  picam2b = Picamera2(1)
  #capture_main = {"size": (7455, 6000), "format": "RGB888"} #dual can do #The resolution can't be 9152 x 6944 for dual camera or else we get a crash on a pi5, but it can do lower
  capture_main = {"size": (4624, 3472), "format": "RGB888"} #dual can do #The resolution can't be 9152 x 6944 for dual camera or else we get a crash on a pi5, but it can do lower
  capture_config = picam2.create_still_configuration(main=capture_main)
  capture_configb = picam2b.create_still_configuration(main=capture_main)

  picam2.configure(capture_config)
  picam2b.configure(capture_configb)

  camera_settings = load_camera_settings()

  #remove settings that aren't actually in picamera2
  computerName = camera_settings.pop("Name",computerName) #defaults to what is set above if not in the files being read

  num_photosHDR = int(camera_settings.pop("HDR")) #defaults to what is set above if not in the files being read
  exposuretime_width = int(camera_settings.pop("HDR_width"))
  if(num_photosHDR<1 or num_photosHDR==2):
      num_photosHDR=1
      
  if camera_settings:
      picam2.set_controls(camera_settings)
      picam2b.set_controls(camera_settings)

  picam2.start()
  picam2b.start()

  time.sleep(2.1)

  print("cam a and b started and then shut off to set their settings");

  picam2.stop()
  picam2b.stop()
  
  picam2.configure(capture_config)
  picam2b.configure(capture_configb)
  
  #important note, to actually 100% lock down an AWB you need to set ColourGains! (0,0) works well for plain white LEDS
  cgains = 2.25943877696990967, 1.500129925489425659
  picam2.set_controls({"ColourGains": cgains})
  picam2b.set_controls({"ColourGains": cgains})
  
  time.sleep(.5)


  # Generate graycode patterns using the structured_light module
  graycode_patterns, whiteimg,blackimg = generate_graycode(projector_res)
  #white_and_black_patterns= generate_shadowmasks
  gray_andTakePhoto()






  picam2.stop()
  picam2b.stop()
