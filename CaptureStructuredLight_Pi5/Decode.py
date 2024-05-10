import os
import cv2
import numpy as np

def load_images_from_directory(directory):
  """
  Loads images from a directory with specific subfolder structure.

  Args:
      directory (str): Path to the directory containing images.

  Returns:
      tuple: A tuple containing two lists:
          - captured_images (list): A list of lists containing loaded images from folders "a" and "b".
          - white_and_black_images (list): A list of lists containing white and black shadow mask images from folders "a" and "b".
  """

  captured_images = [[], []]  # List of lists for captured images from "a" and "b"
  white_and_black_images = [[], []]  # List of lists for white and black shadow masks from "a" and "b"

  # Check if directory exists
  if not os.path.isdir(directory):
    print("Error: Directory", directory, "does not exist.")
    return captured_images, white_and_black_images

  # Check for subfolders "a" and "b"
  a_folder = os.path.join(directory, "a")
  b_folder = os.path.join(directory, "b")
  if not os.path.isdir(a_folder) or not os.path.isdir(b_folder):
    print("Error: Missing folders 'a' or 'b' inside", directory)
    return captured_images, white_and_black_images

  # Load captured images (CamA_X.jpg and CamB_X.jpg)
  for folder, image_list in zip([a_folder, b_folder], captured_images):
    for filename in os.listdir(folder):
      if filename.startswith("CamA") or filename.startswith("CamB"):
        if filename.endswith(".jpg"):
          image_path = os.path.join(folder, filename)
          image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
          if image is not None:
            image_list.append(image)
          else:
            print("Error: Failed to load image", image_path)

  # Load White and Black images (CamA_WB_0.jpg and CamA_WB_1.jpg)
  for folder, image_list in zip([a_folder, b_folder], white_and_black_images):
    for filename in os.listdir(folder):
      if filename.startswith("CamA_WB") or filename.startswith("CamB_WB"):
        if filename.endswith(".jpg"):
          image_path = os.path.join(folder, filename)
          image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
          if image is not None:
            image_list.append(image)
          else:
            print("Error: Failed to load image", image_path)

  return captured_images, white_and_black_images

def compute_shadow_masks(captured_images, white_and_black_images, black_threshold):
  """
  Computes shadow masks for captured images using white and black balance images.

  Args:
      captured_images (list): A list of lists containing captured images from folders "a" and "b".
      white_and_black_images (list): A list of lists containing white and black shadow mask images from folders "a" and "b".
      black_threshold (float): Threshold for shadow detection based on intensity difference.

  Returns:
      list: A list of lists containing computed shadow masks for folders "a" and "b".
  """

  shadow_masks = []
  for i, (cam_a_images, cam_b_images) in enumerate(zip(captured_images, white_and_black_images)):
    # Check if white and black balance images are available
    if len(cam_a_images) < 2 or len(cam_b_images) < 2:
      print(f"Error: Missing white/black balance images for folder {chr(ord('a')+i)}")
      continue

    # Get camera dimensions from the first White and Black image
    cam_rows, cam_cols = white_and_black_images[i][0].shape[:2]

    # Create shadow masks for camera A and B
    shadow_mask_a = white_and_black_images[i][0].copy()
    shadow_mask_b = white_and_black_images[i][1].copy()
    shadow_mask_a.fill(100)  # Set initial value (can be adjusted)
    shadow_mask_b.fill(100)

    # Loop through each pixel in captured images
    for j in range(cam_rows):
      for k in range(cam_cols):
        # Get intensity values from white and black balance images
        white_val = np.int16(white_and_black_images[i][0][j, k])  # Access intensity without the extra [0]
        black_val = np.int16(white_and_black_images[i][1][j, k])  # Access intensity without the extra [0]

        # Check for shadow based on intensity difference and threshold
        if abs(white_val - black_val) > black_threshold:
          shadow_mask_a[j, k] = 255  # Set pixel to white (shadow)
          shadow_mask_b[j, k] = 255
        else:
          shadow_mask_a[j, k] = 0  # Set pixel to black (no shadow)
          shadow_mask_b[j, k] = 0

    shadow_masks.append([shadow_mask_a, shadow_mask_b])

  return shadow_masks

def display_shadow_masks(shadow_masks, wait_time=0):
  """
  Displays shadow mask images for each folder with a specified wait time.

  Args:
      shadow_masks (list): A list of lists containing computed shadow masks for folders "a" and "b".
      wait_time (int, optional): Time (in milliseconds) to display each image. Defaults to 3000 (3 seconds).
  """
  for i, mask_set in enumerate(shadow_masks):
    cv2.imshow(f"Shadow Mask A - Folder {chr(ord('a')+i)}", mask_set[0])
    cv2.waitKey(wait_time)
    cv2.destroyAllWindows()

    cv2.imshow(f"Shadow Mask B - Folder {chr(ord('a')+i)}", mask_set[1])
    cv2.waitKey(wait_time)
    cv2.destroyAllWindows()
    
def save_shadow_masks(shadow_masks, directory):
  """
  Saves shadow mask images to the "decoded" folder within the specified directory.

  Args:
      shadow_masks (list): A list of lists containing computed shadow masks for folders "a" and "b".
      directory (str): Path to the main scan directory.
  """
  decoded_folder = os.path.join(directory, "decoded")
  
  # Create the decoded folder if it doesn't exist
  if not os.path.isdir(decoded_folder):
    os.makedirs(decoded_folder)

  for i, mask_set in enumerate(shadow_masks):
    # Generate filenames for shadow masks
    shadow_mask_a_filename = os.path.join(decoded_folder, f"ShadowMask_{chr(ord('a')+i)}.jpg")
    #shadow_mask_b_filename = os.path.join(decoded_folder, f"ShadowMaskB_Folder{chr(ord('a')+i)}.jpg")

    # Save shadow masks to disk
    cv2.imwrite(shadow_mask_a_filename, mask_set[0])
    #cv2.imwrite(shadow_mask_b_filename, mask_set[1])

def save_decodedImgs(imgA,imgB, directory):

  """

  """
  decoded_folder = os.path.join(directory, "decoded")
  
  # Create the decoded folder if it doesn't exist
  if not os.path.isdir(decoded_folder):
    os.makedirs(decoded_folder)

  for i, mask_set in enumerate(shadow_masks):
    # Generate filenames for shadow masks
    a_filename = os.path.join(decoded_folder, f"decodedA_{chr(ord('a')+i)}.jpg")
    b_filename = os.path.join(decoded_folder, f"decodedB_{chr(ord('a')+i)}.jpg")

    # Save shadow masks to disk
    cv2.imwrite(a_filename,imgA)
    cv2.imwrite(b_filename, imgB)


def reconstruct_projector_pixels(captured_images_gray, shadow_masks, camera_index, 
                                 projector_res_divider, projector_pix_width, projector_pix_height,
                                 get_proj_pixel_func, output_image=None):
  """
  Reconstructs projector pixel locations for a given camera.

  Args:
      captured_images_gray (list): List of grayscale images from the camera.
      shadow_masks (list): List of shadow masks for the camera (same length as captured_images_gray).
      camera_index (int): Index of the camera (0 or 1).
      projector_res_divider (float): Scaling factor for projector resolution.
      projector_pix_width (int): Projector image width.
      projector_pix_height (int): Projector image height.
      get_proj_pixel_func (function): Function that calculates projector pixel location.
      output_image (numpy.ndarray, optional): Optional output image to store reconstructed pixels.

  Returns:
      list, list, list: Lists containing camera X, Y coordinates and corresponding projector X, Y coordinates.
  """
  cam_rows, cam_cols = captured_images_gray[0].shape
  camera_x, camera_y, projector_x, projector_y = [], [], [], []

  for j in range(cam_rows):
    for i in range(cam_cols):
      # Check shadow mask before processing
      if shadow_masks[j, i] > 0:
        proj_pixel = np.zeros((2,), dtype=np.float32)  # Initialize projector pixel coordinates
                #cv.structured_light.GrayCodePattern.getProjPixel(	
        error = get_proj_pixel_func(captured_images_gray, i, j, proj_pixel)

        if error:
          # Handle error (unidentified pixel)
          if output_image is not None:
            output_image[j, i] = [255, 255, 255, 255]  # Set error color (white)
          continue

        # Calculate and store coordinates
        camera_x.append(i)
        camera_y.append(j)
        projector_x.append(int(proj_pixel[0] * projector_res_divider / projector_pix_width))
        projector_y.append(int(proj_pixel[1] * projector_res_divider / projector_pix_height))

        if output_image is not None:
          # Set pixel color based on projector coordinates (adjust color scheme as needed)
          output_image[j, i] = [0, int(proj_pixel[1] * 255 / projector_pix_height), int(proj_pixel[0] * 255 / projector_pix_width), 255]

  return camera_x, camera_y, projector_x, projector_y

def decodeGraycode(photos_cam_a, photos_cam_b, shadow_mask_a, shadow_mask_b, 
                           projector_res_divider, projector_pix_width, projector_pix_height, scan_dir,Ppattern):
  """
  This function processes grayscale images from two cameras (A and B) based on shadow masks.

  Args:
      photos_cam_a: List of OpenCV grayscale images from Camera A.
      photos_cam_b: List of OpenCV grayscale images from Camera B.
      shadow_mask_a: OpenCV image representing the shadow mask for Camera A.
      shadow_mask_b: OpenCV image representing the shadow mask for Camera B.
      projector_res_divider: Divisor for scaling projector pixel coordinates.
      projector_pix_width: Width of the projector image.
      projector_pix_height: Height of the projector image.

  Returns:
      A tuple containing four lists: Xc_a, Yc_a, Xp_a, Yp_a (camera A coordinates) 
      and four lists for camera B (Xc_b, Yc_b, Xp_b, Yp_b).
  """
  pattern_a = np.array(photos_cam_a)
  
  # Lists to store camera and projector pixel coordinates
  Xc_a, Yc_a, Xp_a, Yp_a = [], [], [], []
  Xc_b, Yc_b, Xp_b, Yp_b = [], [], [], []

  for i in range(len(photos_cam_a)):
    # Get the current grayscale image for each camera
    
    # Get the dimensions of the grayscale image
    height, width = photos_cam_a[i].shape

    # Create a new RGB image with the same dimensions and data type (uint8)
    image_a = np.zeros((height, width, 3), dtype=np.uint8)
    image_b = np.zeros((height, width, 3), dtype=np.uint8)

    # Loop through each pixel of the image
    for j in range(image_a.shape[0]):
      for k in range(image_a.shape[1]):

        # Check if the pixel is not shadowed for Camera A
        if shadow_mask_a[j, k] > 0:
          # Implement your logic for finding the corresponding projector pixel using image_a[j, k]
          # This likely involves a function similar to `grayCode.getProjPixel`
          # (replace with your implementation)
          proj_pixel_a = Ppattern.getProjPixel(pattern_a, j, k	)
          #proj_pixel = cv2.structured_light.GrayCodePattern.getProjPixel(pattern_images, x, y)
          if proj_pixel_a is not None:
            # Calculate and store camera and projector pixel coordinates for Camera A
            color_a =[0,255,255]
            #color_a = [0, proj_pixel_a[1] * projector_res_divider/projector_pix_height * 255, 
            #           proj_pixel_a[0] * projector_res_divider / projector_pix_width * 255, 255]
            image_a[j, k]= color_a
            Xc_a.append(k)
            Yc_a.append(j)
            Xp_a.append(proj_pixel_a[0])
            Yp_a.append(proj_pixel_a[1])

        # Similar logic for Camera B (replace placeholders with your implementation)
        if shadow_mask_b[j, k] > 0:
          proj_pixel_b = Ppattern.getProjPixel(photos_cam_b, j, k	)

          if proj_pixel_b is not None:
            color_b =[0,255,255]

            #color_b = [0, proj_pixel_b[1] * projector_res_divider / projector_pix_height * 255, 
            #          proj_pixel_b[0] * projector_res_divider / projector_pix_width * 255, 255]
            image_b[j, k]= color_b

            Xc_b.append(k)
            Yc_b.append(j)
            Xp_b.append(proj_pixel_b[0])
            Yp_b.append(proj_pixel_b[1])
  print("saveimgs")
  save_decodedImgs(imgA,imgB, scan_directory)
  # You can now use the returned lists for further processing or saving the images

  return Xc_a, Yc_a, Xp_a, Yp_a, Xc_b, Yc_b, Xp_b, Yp_b

def create_csv_file(Xc_a, Yc_a, Xp_a, Yp_a, Xc_b, Yc_b, Xp_b, Yp_b, filename="ProjPointsCam"):
  """
  This function creates a CSV file with eight columns from the provided lists.

  Args:
      Xc_a: List of camera A X coordinates.
      Yc_a: List of camera A Y coordinates.
      Xp_a: List of projector X coordinates for camera A.
      Yp_a: List of projector Y coordinates for camera A.
      Xc_b: List of camera B X coordinates.
      Yc_b: List of camera B Y coordinates.
      Xp_b: List of projector X coordinates for camera B.
      Yp_b: List of projector Y coordinates for camera B.
      filename: Optional filename for the CSV file (default: "data.csv").
  """

  # Create a list of lists containing the data for each row
  dataA = [
      [Xc_a[i], Yc_a[i], Xp_a[i], Yp_a[i],] 
      for i in range(len(Xc_a))
  ]
  
  dataB = [
      [Xc_b[i], Yc_b[i], Xp_b[i], Yp_b[i]] 
      for i in range(len(Xc_a))
  ]
  # Write the A data to a CSV file
  with open(filename+"A.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Xc", "Yc", "Xp", "Yp"])
    writer.writerows(dataA)
    
  # Write the B data to a CSV file
  with open(filename+"B.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Xc", "Yc", "Xp", "Yp"])
    writer.writerows(dataB)
  print(f"CSV file created: {filename}")


#~~~~~~~~~USING THE CODE~~~~~~~~~~~~

# Load all the Images
scan_directory = "/home/pi/Desktop/Mothbox/scripts/StructuredLight/Scan_2024-04-25_16-15-53"
captured_images, white_and_black_images = load_images_from_directory(scan_directory)

if captured_images and white_and_black_images:
  print("Successfully loaded images:")
  print("Captured Images:")
  for i, image_set in enumerate(captured_images):
    print(f"\tFolder {chr(ord('a')+i)}:", len(image_set))
  print("White and Black Images:")
  for i, image_set in enumerate(white_and_black_images):
    print(f"\tFolder {chr(ord('a')+i)}:", len(image_set))
else:
  print("Failed to load images. See error messages for details.")
  
  
# Decode the images
## Compute Shadow Masks
blackThreshold = 70  # 3D_underworld default value is 40
whiteThreshold = 50   # 3D_underworld default value is 5
print("starting to compute Shadow Masks")

shadow_masks = compute_shadow_masks(captured_images, white_and_black_images, blackThreshold)

if shadow_masks:
  print("Successfully computed shadow masks.")
  #display_shadow_masks(shadow_masks)
  save_shadow_masks(shadow_masks, scan_directory)
  for i, mask_set in enumerate(shadow_masks):
    print(f"\tFolder {chr(ord('a')+i)} masks:")
    print(f"\t\tShape of shadow mask A:", mask_set[0].shape)
    print(f"\t\tShape of shadow mask B:", mask_set[1].shape)
else:
  print("Failed to compute shadow masks. See error messages for details.")

print("starting to decode Graycode")

#Next decode the graycode


projector_pix_width = 1920
projector_pix_height= 1080
projector_divide_res = 12
resolution=[int(projector_pix_width/projector_divide_res),int(projector_pix_height/projector_divide_res)]
pattern = cv2.structured_light.GrayCodePattern.create(width=resolution[0], height=resolution[1])

Xc_a, Yc_a, Xp_a, Yp_a, Xc_b, Yc_b, Xp_b, Yp_b = decodeGraycode(captured_images[0], captured_images[1], white_and_black_images[0][0], white_and_black_images[1][0], 
                           projector_divide_res, projector_pix_width, projector_pix_height, scan_directory, pattern)

create_csv_file(Xc_a, Yc_a, Xp_a, Yp_a,Xc_a, Yc_a, Xp_a, Yp_a, Xc_b, Yc_b, Xp_b, Yp_b)
