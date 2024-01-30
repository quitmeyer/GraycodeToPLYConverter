# Graycode Scan 

## Unity Graycode Scan

### brio notes
When using two brio cameras, they historically have problems outputting the full 4k they are suposed to. One workaround has been to plug them both into a USB C hub, and plug it into a USBC thunderbolt port

another person told me their workaround was to download some sort of program that people use for twitch and only fans that optimizes the cameras for streaming

## Start Unity Scan
anyway first step is to connect the cameras and then (using something like amcap or the logitune app
* lock the exposure (i use -8) (too short and it breaks a projector into rainbows, too long and it overexposes)
* lock the focus (i just locked em down to 0 and they work fine, kind of like an infinity focus)

Then launch the Unity Structured light program

click the "Canvas SL"
This lets you look at all the options for the structured light program

![image](https://github.com/quitmeyer/GraycodeToPLYConverter/assets/742627/6288d800-5e4b-44c0-8030-89454eb84ba8)

* set projector width and height
* set a resolution divider for the projector (sometimes your projector will project too narrow of lines for your camera to see)
* select the correct ports for the cameras. Cam A is usually the right camera (when looking from behind the camera pair). Cam A will display on the top in the canvas view

## Run the Scan
Next you can hit the big play button and start the program running.

Switch to your game window and bring it to the projector's display

Press F11 to activate the FULLSCREEN plugin and make the game window entirely fullscreen without any bars

When it first starts up, your screen will show you previews of both cameras.  You can use this to double check cam A is on top
## Scanning
* press A to activate automatic mode
* then press spacebar to start the automatic scan
* After the scan has stopped, first press P to process all the images (this will take a minute or so)
* After that is finished, you can press S to save all the images to your harddrive (this will also take a minute or so)

Currently files get saved to

C:\Users\andre\Desktop\Glowcake Hoss\Structured Light\Graycode


# Directory Structure

To help with colmap processing, your data should eventually be in a directory/file naming structure like this
```
├── project_scan_date
│   ├── pg
│   │   ├── model
│   │   ├── img
│   │   │   ├── a
│   │   │   ├── b
│   │   │   ├── projector
│   │   │   │   ├──white3840.png (a blank png with the width and height of the projector you used)
│   │   ├── a
│   │   ├── b
│   │   ├── projector

├── sl
│   ├── a
│   ├── b
│   ├── decoded
│   │   ├── ProjPixMatA.png
│   │   ├── ProjPixMatB.png
│   │   ├── ProjPointsCamA.CSV
│   │   ├── ProjPointsCamB.CSV
```

# Colmap Processing


## Fresh Full Scan (Calibrate Camera and Projector Intrinsics)
If the cameras and projector are substantially different from other scans, we need to do a really good full scan and get the camera and projector intrinsics from this. Later if we already have the intrinsics for the cameras and projectors, we can skip to the "Minimal Scan" option, where we do not need to do a long format photogrammetry scan.



### Photogrammetry

##### Stereo Capture with Brio cameras

This runs a program that captures stereo, high res photos of a scene. Once started it should make a beep/ding sound every time the computer captures an image. The output shows how many photos have been taken. This is useful for a photogrammetry scan of a scene.

#### Run the Capture
Run example_aruco_

Calibrate_camera_dualcamera_photos_cpp

Collect a bunch of photos
It takes a photo about every two seconds and should make your computer ding

#### Save the photos
It takes a while to save the photos because they are large. So we save them all after the program have captured photos to memory.

Press "s" or "esc" to start saving
Files get saved to
C:\Users\andre\Desktop\Glowcake Hoss\Scans




### Colmap Full Scan Process Photogrammetry
You need to go into the PG photogrammetry folders called A and B and find the photos taken in the position of the canonical cameras for the Structured light ((usually this is your photo called camA_im0.png)) and rename two of those images to be copies of the canonical camera images
* 
* a/CamA_WB_1.png
* And
* b/CamB_WB_1.png

in colmap create a new database with an images folder that points to where the a and b subfolders are
* click "Feature Extraction"
* choose "shared intrinsics per subfolder
* Choose the camera model you are using (currently we are using Radial)
* run feature extraction
* next run Match features



5) Run the goocher to tie things together
* set up your command line parameters for the goocher by editing the runGraycodeGoocher.bat
* 

7) Run Colmap AGAIN

 Make sure that “ignore two-view geometries” is UNCHECKED in the reconstruction options

Make sure to click “refine principle point” in reconstruction options

(though maybe just refine focal distance? This actually seems a lot better!)

Export the reconstructed model (it took 26 minutes to reconstruct my last scan)

WATCH OUT FOR CAMERA SWITCHING!

DOUBLE CHECK FEATURES on CAMA_WB1 

7)

# Manipulating the Mesh

## Import into Blender

There is a script you can find online that will import colmap scenes into blender. When importing, in the import options, make sure to click “import point cloud as mesh” so that you get something that can actually be rendered in blender (you can’t render pointclouds in blender without a headache)

You also need to import the revopoint scan

8) do distortion in blender

https://blender.stackexchange.com/questions/181062/how-to-apply-custom-lens-distortion-parameters-to-a-rendered-image

(Other info [https://blender.stackexchange.com/questions/121841/customize-blender-camera-distortion](https://blender.stackexchange.com/questions/121841/customize-blender-camera-distortion)

insert camera params. Cam params from colmap

[https://colmap.github.io/cameras.html](https://colmap.github.io/cameras.html)

[https://github.com/colmap/colmap/blob/main/src/colmap/sensor/models.h](https://github.com/colmap/colmap/blob/main/src/colmap/sensor/models.h)

eg. ocv is 

//	fx, fy, cx, cy, k1, k2, p1, p2

//////


## New command line tests

Blank out the lines underneath Cam A and Cam B (don't d delete leave blank

Gotta run Clear points!

colmap.bat point_triangulator --database_path C:\Users\andre\Desktop\colmapPy\demoGrayCode\vs_dec5_SimplePinhole_0_new.db --image_path C:\Users\andre\Desktop\colmapPy\demoGrayCode\img --input_path C:\Users\andre\Desktop\colmapPy\demoGrayCode\VSdec5_Gooched_simplepinManEdit --output_path C:\Users\andre\Desktop\colmapPy\demoGrayCode\VSdec5_tri --Mapper.tri_ignore_two_view_tracks 1 --clear_points 1

reduce track length to 2!

colmap.bat mapper --database_path C:\Users\andre\Desktop\colmapPy\demoGrayCode\vs_dec5_SimplePinhole_0_new.db --image_path C:\Users\andre\Desktop\colmapPy\demoGrayCode\img --input_path C:\Users\andre\Desktop\colmapPy\demoGrayCode\VSdec5_tri --output_path C:\Users\andre\Desktop\colmapPy\demoGrayCode\VSdec5_tri_map --Mapper.tri_ignore_two_view_tracks 1
