# Graycode Scan 

## Unity Graycode Scan

### Unity Project - Open CV
the unity project relies on an opencv library
download it here
https://drive.google.com/file/d/1ZIko9aMS_3S2IplblNl7VXLaJkPsjrMt/view?usp=sharing
unzip it to your Assets folder in the unity project
![image](https://github.com/quitmeyer/GraycodeToPLYConverter/assets/742627/21b4c18b-2dba-4d77-9f7e-4ac0ecd625e8)


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
└── GlowcakeDatabaseGoocher.py
└── runGrayCodeGoocher.bat
├── project_scan_date (eg. Scan_2024-01-30T22_31Z)
│   ├── pg
│   │   ├── models
│   │   ├── img
│   │   │   ├── a
│   │   │   │   └── "CamA_canon"+".png"
│   │   │   ├── b
│   │   │   │   └── "CamB_canon"+".png"
│   │   │   ├── projector
│   │   │   │   └── white3840.png (a blank png with the width and height of the projector you used)
│   └── database.db
│   ├── sl
│   │   ├── a
│   │   │   └── "CamA_" + graycodeseriesnum + ".png"
│   │   ├── b
│   │   │   └── "CamB_" + graycodeseriesnum + ".png"
│   │   ├── decoded
│   │   │   └── ProjPixMatA.png
│   │   │   └── ProjPixMatB.png
│   │   │   └── ProjPointsCamA.CSV
│   │   │   └── ProjPointsCamB.CSV
```

# Colmap Processing

## Simple Scan Process
After doing a structrured light scan, copy your scan folder to where it sits at the same level at the GraycodeGoocher.py

## Colmap Matching (round 1)
Open Colmap
Create new project
make a new database in the 
project->pg folder
call it "database.db"
set the images to point to the project->pg->img folder
![image](https://github.com/quitmeyer/GraycodeToPLYConverter/assets/742627/7274f184-5800-4d96-99df-79616aaabccb)

click "Feature Extraction"
choose : Camera mode
Choose : "Shared per sub-folder"
click "Extract"
close that dialog

now click "Feature Matching"
click "Run"
close the dialog


# Graycode Matches Injector (Goocher)

Run the goocher to tie things together

## Setup Goocher Command Line Arguments
You might need to customize the goocher to the projector you are using
Here is a list of the arguments that the Graycode goocher contains and their defaults
you can edit the rungraycodegoocher.bat file to change these
```
    parser.add_argument("--project", default="glowcake_01_2024")
    parser.add_argument("--db", default="pg/database.db")
    parser.add_argument("--camAPoints", default="sl/ProjPointsCamA.CSV")
    parser.add_argument("--camBPoints", default="sl/ProjPointsCamB.CSV")
    parser.add_argument("--projWidth", default="3840")
    parser.add_argument("--projHeight", default="2160")
    parser.add_argument("--projImage", default="white3840.png")
    parser.add_argument("--subSample", default="1")
    parser.add_argument("--camModel", default="Radial")

```
it should tell you you are finished graycode gooching, and you should have a new database file in your project!

## Run Colmap AGAIN

back in colmap click edit project

choose the NEW database file you made with the goocher

![image](https://github.com/quitmeyer/GraycodeToPLYConverter/assets/742627/89ee8569-74ae-45d4-a257-e9f48d0d6147)
click save

click the database button
manually edit the cameras intrinsics to match what you collected earlier

![image](https://github.com/quitmeyer/GraycodeToPLYConverter/assets/742627/305fe36c-8922-4790-bf24-0df2870ce36b)

go to Reconstruction->reconstruction options
click "Triangulation..."
uncheck "ignore two-view"
go to "Bundle adjust..."
uncheck all three "refine..."



Hit the big play button, and then in a minute or two you should have your model!

![image](https://github.com/quitmeyer/GraycodeToPLYConverter/assets/742627/0fbd0614-25f8-436f-b7f4-d7aa026eab01)

click File->export model as Text
save your model in your pg->models folder


# Manipulating the Mesh

## Import into Blender

There is a script you can find online that will import colmap scenes into blender. When importing, in the import options, make sure to click “import point cloud as mesh” so that you get something that can actually be rendered in blender (you can’t render pointclouds in blender without a headache)

File - > Import - > Colamp
make sure to click "import points as mesh object"

![image](https://github.com/quitmeyer/GraycodeToPLYConverter/assets/742627/93ae93d6-8e9f-4ad8-b882-0fe5b1155d1a)

It should import and you should see your scan

next save the whole blender project
![image](https://github.com/quitmeyer/GraycodeToPLYConverter/assets/742627/6f4da6f5-b563-4c2f-b1ab-1f49ee6a7842)

## Import into Unity
Open your unity project
click "Import New Asset"
![image](https://github.com/quitmeyer/GraycodeToPLYConverter/assets/742627/2d952038-be93-42e5-99c2-f5228521fbf2)

Choose your blender file(it might take a second to load)

you can drag it into your scene now
![image](https://github.com/quitmeyer/GraycodeToPLYConverter/assets/742627/36f859f0-3207-4dd0-b268-fe95f11b8143)

you need to set your projector camera as your "main camera"
and deactivate any cameras above it

Bring your game window to the live projector screen
Hit F11 to make full screen, all pixels

now you can dress your scene and do as you wish, and the cameras and objects should be aligned!
![image](https://github.com/quitmeyer/GraycodeToPLYConverter/assets/742627/07d1e307-4f81-4f2b-83f8-a5f55b0faacd)




























### (optional for now) do distortion in blender

https://blender.stackexchange.com/questions/181062/how-to-apply-custom-lens-distortion-parameters-to-a-rendered-image

(Other info [https://blender.stackexchange.com/questions/121841/customize-blender-camera-distortion](https://blender.stackexchange.com/questions/121841/customize-blender-camera-distortion)

insert camera params. Cam params from colmap

[https://colmap.github.io/cameras.html](https://colmap.github.io/cameras.html)

[https://github.com/colmap/colmap/blob/main/src/colmap/sensor/models.h](https://github.com/colmap/colmap/blob/main/src/colmap/sensor/models.h)

eg. ocv is 

//	fx, fy, cx, cy, k1, k2, p1, p2

//////


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







# Future Structures for Multiview
This is for a single object being scanned by an arbitrary number of Proj-Cam-Cam triangles

This would be the resulting structure for a scan with two separate structured light scans and knowledge of existing intrinsics

```
└── GlowcakeDatabaseGoocher.py
└── runGrayCodeGoocher.bat
├── project_scan_date (eg. Scan_2024-01-30T22_31Z)
│   ├── pg
│   │   ├── models
│   │   ├── img
│   │   │   ├── t0_a
│   │   │   │   └── "CamA_canon"+".png"
│   │   │   ├── t0_b
│   │   │   │   └── "CamB_canon"+".png"
│   │   │   ├── t1_a
│   │   │   │   └── "CamA_canon"+".png"
│   │   │   ├── t1_b
│   │   │   │   └── "CamB_canon"+".png"
│   │   ├── img_injected
│   │   │   ├── t0_a
│   │   │   │   └── "CamA_canon"+".png"
│   │   │   ├── t0_b
│   │   │   │   └── "CamB_canon"+".png"
│   │   │   ├── t0_projector
│   │   │   │   └── white3840.png (a blank png with the width and height of the projector you used)
│   │   │   ├── t1_a
│   │   │   │   └── "CamA_canon"+".png"
│   │   │   ├── t1_b
│   │   │   │   └── "CamB_canon"+".png"
│   │   │   ├── t1_projector
│   │   │   │   └── white3840.png (a blank png with the width and height of the projector you used)
│   └── database.db
│   └── database_injected.db
│   ├── sl_t0
│   │   └── triangle_info.yaml
│   │   ├── a
│   │   │   └── "CamA_" + graycodeseriesnum + ".png"
│   │   ├── b
│   │   │   └── "CamB_" + graycodeseriesnum + ".png"
│   │   ├── proj
│   │   ├── decoded
│   │   │   └── ProjPixMatA.png
│   │   │   └── ProjPixMatB.png
│   │   │   └── ProjPointsCamA.CSV
│   │   │   └── ProjPointsCamB.CSV
│   ├── sl_t1
│   │   └── triangle_info.yaml
│   │   ├── a
│   │   │   └── "CamA_" + graycodeseriesnum + ".png"
│   │   ├── b
│   │   │   └── "CamB_" + graycodeseriesnum + ".png"
│   │   ├── proj
│   │   ├── decoded
│   │   │   └── ProjPixMatA.png
│   │   │   └── ProjPixMatB.png
│   │   │   └── ProjPointsCamA.CSV
│   │   │   └── ProjPointsCamB.CSV


```

it would have a yaml file in each SL scan that is formatted like this:

```
YAML

# Example YAML file with Light Triangles and Nodes that are projectors or cameras
# Each light triangle has one projector and two cameras

LightTriangle:
  -name: t0
  -a:
    - type: cam
      width: 4096
      height: 2160
      cameramodel: Radial
      intrinsics: 3060.217151, 2010.003211, 980.941691, 0.176927, -0.308681
  -b: 
    - type: cam
      width: 4096
      height: 2160
      cameramodel: Radial
      intrinsics: 3058.494720, 2047.179953, 1098.225191, 0.193613, -0.357522
  -projector: 
    - type: projector
      width: 3840
      height: 2160
      cameramodel: Radial
      intrinsics: 3058.494720, 2047.179953, 1098.225191, 0.193613, -0.357522

```














~~~ old stuff
## New command line tests

Blank out the lines underneath Cam A and Cam B (don't d delete leave blank

Gotta run Clear points!

colmap.bat point_triangulator --database_path C:\Users\andre\Desktop\colmapPy\demoGrayCode\vs_dec5_SimplePinhole_0_new.db --image_path C:\Users\andre\Desktop\colmapPy\demoGrayCode\img --input_path C:\Users\andre\Desktop\colmapPy\demoGrayCode\VSdec5_Gooched_simplepinManEdit --output_path C:\Users\andre\Desktop\colmapPy\demoGrayCode\VSdec5_tri --Mapper.tri_ignore_two_view_tracks 1 --clear_points 1

reduce track length to 2!

colmap.bat mapper --database_path C:\Users\andre\Desktop\colmapPy\demoGrayCode\vs_dec5_SimplePinhole_0_new.db --image_path C:\Users\andre\Desktop\colmapPy\demoGrayCode\img --input_path C:\Users\andre\Desktop\colmapPy\demoGrayCode\VSdec5_tri --output_path C:\Users\andre\Desktop\colmapPy\demoGrayCode\VSdec5_tri_map --Mapper.tri_ignore_two_view_tracks 1
