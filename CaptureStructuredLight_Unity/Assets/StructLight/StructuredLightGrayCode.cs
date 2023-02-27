
using System.Collections.Generic;
using UnityEngine;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityUtils;
using OpenCVForUnity.Structured_lightModule;
using UnityEngine.Windows.WebCam;
using System.Linq;
using System.IO;
using UnityEngine.UI;

public class StructuredLightGrayCode : MonoBehaviour
{
    Params paramz;
    GrayCodePattern grayCode;

    WebCamDevice[] devices;
    WebCamTexture camAwebcam;
    WebCamTexture camBwebcam;


    List<Mat> pattern;
    List<Mat> patternWB;
    List<Mat> photosCam1;
    List<Mat> photosCam2;
    List<Mat> photosCam1WB;//and all white and all black photos
    List<Mat> photosCam2WB;//and all white and all black photos
    public bool saveAllGrayCodes = true;
    Mat white;
    Mat black;

    int blackThreshold = 70;  // 3D_underworld default value is 40
    int whiteThreshold = 50;   // 3D_underworld default value is 5

    /// <summary>
    /// The texture.
    /// </summary>
    Texture2D grayCodeTexture;
    Texture2D displayTexture;


    int pattnum = -1;

    List<int> Xc, Yc, Xp, Yp;

    /// <summary>
    /// Set the width projector for scanning.
    /// </summary>
    [SerializeField, TooltipAttribute("Set the width projector for scanning.")]
    public int projectorPixWidth = 640;

    /// <summary>
    /// Set the height of projector for scanning
    /// </summary>
    [SerializeField, TooltipAttribute("Set the height of projector for scanning.")]
    public int projectorPixHeight = 108;

    int numOfColImgs;
    int numOfRowImgs;


    //Webcam stuff

    /// <summary>
    /// Set the name of the device to use.
    /// </summary>
    [SerializeField, TooltipAttribute("Set the name of the device to use.")]
    public string requestedDeviceName = null;

    /// <summary>
    /// Set FPS of WebCamTexture.
    /// </summary>
    [SerializeField, TooltipAttribute("Set FPS of WebCamTexture.")]
    public int requestedFPS = 30;

    /// <summary>
    /// Set whether to use the front facing camera.
    /// </summary>
    [SerializeField, TooltipAttribute("Set whether to use the front facing camera.")]
    public bool requestedIsFrontFacing = false;

    /// <summary>
    /// The webcam texture.
    /// </summary>
    WebCamTexture webCamTexture;

    /// <summary>
    /// The webcam device.
    /// </summary>
    WebCamDevice webCamDevice;

    bool Cam1FinishedProcessing = false;
    bool AllWBcaptured = false;
    bool AllGraycodeCaptured = false;
    bool firstframe = true;

    //Webcam stuff
    [SerializeField, TooltipAttribute("Choose which camera to use for Cam A.")]
    public int camA_port = 0;
    [SerializeField, TooltipAttribute("Choose which camera to use for Cam B.")]
    public int camB_port = 2;


    PhotoCapture photo1CaptureObject = null;
    Texture2D photo1TargetTexture = null;
    PhotoCapture photo2CaptureObject = null;
    Texture2D photo2TargetTexture = null;
    // Start is called before the first frame update

    GameObject camAdisplay;
    GameObject camBdisplay;

    Mat ShadowMask;
    Mat ProjPixMat;
    void Start()
    {


        camAdisplay = GameObject.Find("CamAImage");
        camBdisplay = GameObject.Find("CamBImage");
        //Set Cursor to not be visible
        // Cursor.visible = false;
        //Cursor.lockState = CursorLockMode.Locked;

        //Screen.SetResolution(projectorPixWidth, projectorPixHeight, true);

        pattern = new List<Mat>();
        patternWB = new List<Mat>();
        photosCam1 = new List<Mat>();
        photosCam2 = new List<Mat>();
        photosCam1WB = new List<Mat>();
        photosCam2WB = new List<Mat>();

        //coordinates of projected pixels
        //Xc is camera pixel, Xp is projector location
        Xc = new List<int>();
        Yc = new List<int>();
        Xp = new List<int>();
        Yp = new List<int>();

        black = new Mat();
        white = new Mat();
        ShadowMask = new Mat();
        ProjPixMat = new Mat();

        paramz = new Params();

        Debug.Log("Projector Size W = " + projectorPixWidth + " H  = " + projectorPixHeight);

        grayCode = GrayCodePattern.create(projectorPixWidth, projectorPixHeight);

        grayCode.generate(pattern);

        Debug.Log("pattern size (OCV) = " + pattern.Count);
        //HMM there is no function to get num of rows vs columns grayCode.get
        grayCode.getImagesForShadowMasks(black, white);
        patternWB.Add(white);
        patternWB.Add(black);

        Debug.Log("patternWB size = " + patternWB.Count);

        //SaveAllGrayCodetoPNG();

        /*Link the Raw Image to the Graycode Texture*
        grayCodeTexture = new Texture2D(projectorPixWidth, projectorPixHeight);//, TextureFormat.ARGB32, false);
        //GetComponent<RawImage>().material.mainTexture = grayCodeTexture;

        //Display the first gray code of Pattern 0
        //Update to the next pattern to display
        Mat thepattern = new Mat();
        // pattern[pattnum].convertTo(thepattern, CvType.CV_8UC1);

        //Make Unity Display The first graycode texture
        Utils.matToTexture2D(pattern[pattnum], grayCodeTexture); // Opencv Way

        displayMat(pattern[pattnum]);
        Debug.Log("Currently Showing Graycode Pattern: " + pattnum);
        /*  */
        InitializeCamera();

    }


    // Update is called once per frame
    void Update()
    {


        //Update the Graycode once the camera finished processing
        if (Cam1FinishedProcessing)
        {


            if (pattnum < pattern.Count - 1 && pattnum > -2)
            {
                //Update to the next pattern to display
                pattnum++;
                //Make Unity Display The next graycode texture
                //Utils.matToTexture2D(pattern[pattnum], grayCodeTexture); // Opencv Way
                DisplayMat(pattern[pattnum]);                                                          //  gameObject.GetComponent<CanvasRenderer>().SetTexture(grayCodeTexture);
                Debug.Log("Currently Showing Graycode Pattern: " + pattnum);
            }
            if (AllGraycodeCaptured && !AllWBcaptured)
            {
                pattnum++;
                //Update Graycode Pattern
                //Utils.matToTexture2D(patternWB[pattnum - pattern.Count], grayCodeTexture);
                DisplayMat(patternWB[pattnum - pattern.Count]);
                //  gameObject.GetComponent<CanvasRenderer>().SetTexture(grayCodeTexture);

                // gameObject.GetComponent<CanvasRenderer>().SetTexture(grayCodeTexture);
                Debug.Log("Showing WB Pattern: " + pattnum);

                if (pattnum - pattern.Count == 2)
                {
                    AllWBcaptured = true;
                }
            }

            Cam1FinishedProcessing = false;
        }

        //Take photo of graycode
        if (Input.GetKeyDown(KeyCode.Space))
        {
            if (AllWBcaptured)
            {
                Debug.Log("Finished capturing all images, press P to process, Press S to save all cap img to disk");
            }
            else
            {
                //Capture the Graycodes being displayed
                if (pattnum < pattern.Count - 1 && pattnum > -2)
                {
                    //Snag the Photo of the previous pattern being displayed
                    GrabGrayCodePhoto(pattnum);
                }
                if (pattnum == pattern.Count - 1)
                {
                    //Save the final Graycode image, and toggle to WB images
                    GrabGrayCodePhoto(pattnum);
                    AllGraycodeCaptured = true;
                }
                //Run the Black and White Image Captures
                if (pattnum > pattern.Count - 1 && AllGraycodeCaptured && !AllWBcaptured)
                {
                    Debug.Log("All Graycode Captured, Doing WB thresholds");

                    if (pattnum < pattern.Count + 1)
                    {
                        GrabWBPhoto(pattnum, "white");

                    }
                    else
                    {
                        GrabWBPhoto(pattnum, "black");
                        AllWBcaptured = true;
                    }
                }
            }

        }

        if (Input.GetKeyDown(KeyCode.S))
        {
            Debug.Log("Saving ALL Camera imgs... ");

            //Save all Captured Textures to Disk
            for (int i = 0; i < photosCam1.Count; i++)
            {
                Texture2D theTex = new Texture2D(photo1TargetTexture.width, photo1TargetTexture.height);

                //Save image to disk
                Utils.matToTexture2D(photosCam1[i], theTex);

                byte[] bytes = theTex.EncodeToPNG();
                string filePath = "Graycode/Cam1_" +
                   i + ".png";
                File.WriteAllBytes(Application.absoluteURL + filePath, bytes);
                Debug.Log("Saving Camera img PNG: " + filePath);
            }
            for (int i = 0; i < photosCam1WB.Count; i++)
            {
                Texture2D theTex = new Texture2D(photo1TargetTexture.width, photo1TargetTexture.height);

                //Save image to disk
                Utils.matToTexture2D(photosCam1WB[i], theTex);

                byte[] bytes = theTex.EncodeToPNG();
                string filePath = "Graycode/Cam1_WB_" +
                   i + ".png";
                File.WriteAllBytes(Application.absoluteURL + filePath, bytes);
                Debug.Log("Saving Camera img PNG: " + filePath);
            }
            Debug.Log("Saved all graycode and WB captures ");
        }

        if (Input.GetKeyDown(KeyCode.L))
        {
            Debug.Log("Loading Previous Graycode Captures");
            //Load a photo for every graycode generated

            //first clear all any photos that may have been stored
            photosCam1.Clear();
            photosCam1WB.Clear();

            for (int i = 0; i < grayCode.getNumberOfPatternImages(); i++)
            {
                // Create an array of file paths from which to choose
                //Below is an example to read all PNG's from a folder
                //folderPath = Application.streamingAssetsPath + "/NameImages/GameNames/";  //Get path of folder
                //       filePaths = Directory.GetFiles(folderPath, "*.png"); // Get all files of type .png in this folder

                string filePath = "Graycode/Cam1_" +
                  i + ".png";
                //Converts desired path into byte array
                byte[] pngBytes = File.ReadAllBytes(filePath);

                //Creates texture and loads byte array data to create image
                Texture2D tex = new Texture2D(photo1TargetTexture.width, photo1TargetTexture.height);
                tex.LoadImage(pngBytes);
                //TODO make these mats reconfigure to the size of the graycode detected
                Mat mat = new Mat(photo1TargetTexture.height, photo1TargetTexture.width, CvType.CV_8UC4);
                Utils.texture2DToMat(tex, mat);
                photosCam1.Add(mat);


            }
            for (int i = 0; i < 2; i++)
            {
                string filePath = "Graycode/Cam1_WB_" +
                  i + ".png";
                //Converts desired path into byte array
                byte[] pngBytes = File.ReadAllBytes(filePath);

                //Creates texture and loads byte array data to create image
                Texture2D tex = new Texture2D(photo1TargetTexture.width, photo1TargetTexture.height);
                tex.LoadImage(pngBytes);
                Mat mat = new Mat(photo1TargetTexture.height, photo1TargetTexture.width, CvType.CV_8UC4);
                Utils.texture2DToMat(tex, mat);
                photosCam1WB.Add(mat);

            }

            Debug.Log("Loading Graycode Captures Total Graycodes = " + photosCam1.Count + " and total WB captures = " + photosCam1WB.Count);

            //Display one of the captures
        
            DisplayMat(photosCam1WB[0]);


        }
        if (Input.GetKeyDown(KeyCode.W))
        {
            // gameObject.GetComponent<CanvasRenderer>().SetTexture(webCamTexture);
            Texture2D texturego = new Texture2D(photo1TargetTexture.width, photo1TargetTexture.height);

            //Choose a rand photo to look at
            int photoRand = Random.Range(0, photosCam1.Count - 1);
            Debug.Log("photo to check " + photoRand);

            Imgproc.threshold(photosCam1[photoRand], ShadowMask, 50, 255, Imgproc.THRESH_BINARY);

            Utils.matToTexture2D(ShadowMask, texturego);

            texturego.Apply(); //OMGFG IT JUST NEEDED APPLY!!! WTF
            // Utils.matToTexture2D(ShadowMask, texturego);
            //Utils.matToTexture2D(ShadowMask, grayCodeTexture);

            //gameObject.GetComponent<CanvasRenderer>().SetTexture(texturego);
            //grayCodeTexture = texturego;
            //gameObject.GetComponent<RawImage>().material.mainTexture = texturego;
            gameObject.GetComponent<CanvasRenderer>().SetTexture(texturego);
            //gameObject.GetComponent<RawImage>().mainTexture.Equals(texturego);
            //SetTexture(texturego);
            //GetComponent<RawImage>().material.mainTexture = texturego;



            byte[] bytes = texturego.EncodeToPNG();
            string filePath = "Graycode/RAND_" +
                ".png";
            File.WriteAllBytes(Application.absoluteURL + filePath, bytes);
            Debug.Log("Saving RAND img PNG: " + filePath);

        }
        if (Input.GetKeyDown(KeyCode.P))
        {
            Debug.Log("Processing the Graycode... ");

            ComputeShadowMasks();
            ReadGrayCodeImages();


        }

    }
    void SaveAllGrayCodetoPNG()
    {
        //Debug Save all patterns as files
        if (saveAllGrayCodes == true)
        {

            for (int i = 0; i < pattern.Count; i++)
            {
                Texture2D GraycodeDebugTex = new Texture2D(projectorPixWidth, projectorPixHeight);
                Utils.matToTexture2D(pattern[i], GraycodeDebugTex); // Opencv Way

                byte[] bytes = GraycodeDebugTex.EncodeToPNG();
                string filePath = "Graycode/Graycode_" +
                    i + ".png";
                File.WriteAllBytes(Application.absoluteURL + filePath, bytes);
                Debug.Log("Saving Graycode PNG: " + filePath);
            }
        }

    }

    void InitializeCamera()
    {
        //webcam stuff
        // webCamTexture = new WebCamTexture();
        // webCamTexture.Play();
        //colors = new Color32[webCamTexture.width * webCamTexture.height];

        //Setup Webcams
        devices = WebCamTexture.devices;
        for (int i = 0; i < devices.Length; i++) // How many webcams we got?
            Debug.Log("Webcam " + i + " name = " + devices[i].name);


        //Photo capture stuff
        Debug.Log("Camera resolution" + PhotoCapture.SupportedResolutions.OrderByDescending((res) => res.width * res.height).First());
     
        Resolution cameraResolution = PhotoCapture.SupportedResolutions.OrderByDescending((res) => res.width * res.height).First();
        List<Resolution> allreslist = PhotoCapture.SupportedResolutions.OrderByDescending((res) => res.width * res.height).ToList<Resolution>();

        for (int i=0;i< allreslist.Count; i++)
        {
            Debug.Log("What them resss" + allreslist[i]);

        }

        camAwebcam = new WebCamTexture(devices[camA_port].name, cameraResolution.width, cameraResolution.height, 60);
        camBwebcam = new WebCamTexture(devices[camB_port].name, cameraResolution.width, cameraResolution.height, 60);
        Debug.Log("Cam A FPS " + camAwebcam.requestedFPS);


        cameraResolution = allreslist[allreslist.Count-1];

        photo1TargetTexture = new Texture2D(cameraResolution.width, cameraResolution.height);
       displayTexture= new Texture2D(photo1TargetTexture.width, photo1TargetTexture.height);
        // Create a PhotoCapture object
        PhotoCapture.CreateAsync(false, delegate (PhotoCapture captureObject)
        {
            photo1CaptureObject = captureObject;
            CameraParameters cameraParameters = new CameraParameters();
            cameraParameters.hologramOpacity = 0.0f;
            cameraParameters.cameraResolutionWidth = cameraResolution.width;
            cameraParameters.cameraResolutionHeight = cameraResolution.height;
            cameraParameters.pixelFormat = CapturePixelFormat.BGRA32;

            // Activate the camera
            photo1CaptureObject.StartPhotoModeAsync(cameraParameters, delegate (PhotoCapture.PhotoCaptureResult result)
            {
                // Take an initial picture
                photo1CaptureObject.TakePhotoAsync(OnCapturedPhotoToMemoryCam1);
            });
        });
    }

    void OnCapturedPhotoToMemoryCam1(PhotoCapture.PhotoCaptureResult result, PhotoCaptureFrame photoCaptureFrame)
    {
        // Copy the raw image data into our target texture
        photoCaptureFrame.UploadImageDataToTexture(photo1TargetTexture);
        Debug.Log("cam1 to Tex1 ~~~  Captured photo at index: " + pattnum);
        Mat cam1Mat = new Mat(photo1TargetTexture.height, photo1TargetTexture.width, CvType.CV_8UC4);

        Utils.texture2DToMat(photo1TargetTexture, cam1Mat);
        photosCam1.Add(cam1Mat);
        Debug.Log("MatPhoto from cam1 added to photoscam1 at index: " + (photosCam1.Count - 1));

        camAdisplay.GetComponent<Image>().material.mainTexture =  photo1TargetTexture;
        Cam1FinishedProcessing = true;
    }

    void OnCapturedWBPhotoToMemoryCam1(PhotoCapture.PhotoCaptureResult result, PhotoCaptureFrame photoCaptureFrame)
    {
        // Copy the raw image data into our target texture
        photoCaptureFrame.UploadImageDataToTexture(photo1TargetTexture);
        Debug.Log("cam1 to Tex1 ~~~  Captured photo at index: " + pattnum);
        Mat cam1Mat = new Mat(photo1TargetTexture.height, photo1TargetTexture.width, CvType.CV_8UC4);


        Utils.texture2DToMat(photo1TargetTexture, cam1Mat);

        photosCam1WB.Add(cam1Mat);
        Debug.Log("MatPhoto from cam1 added to photoscam1WB at index: " + (photosCam1WB.Count - 1));

        Cam1FinishedProcessing = true;
    }

    void OnCapturedPhotoToDiskCam1(PhotoCapture.PhotoCaptureResult result)
    {
        Cam1FinishedProcessing = true;
        Debug.Log("Saved Cam1 Image to disk at index: " + pattnum);
        //Save image to disk
        /* byte[] bytes = photo1TargetTexture.EncodeToPNG();
         string filePath = "Graycode/Cam1_" +
            index + ".png";
         File.WriteAllBytes(Application.absoluteURL + filePath, bytes);
         Debug.Log("Saving Camera img PNG: " + filePath);
         */
    }

    void OnStoppedPhotoMode(PhotoCapture.PhotoCaptureResult result)
    {
        // Shutdown our photo capture resource
        photo1CaptureObject.Dispose();
        photo1CaptureObject = null;
    }
    void DisplayTexture(Texture2D theTex)
    {

        theTex.Apply(); //OMGFG IT JUST NEEDED APPLY!!! WTF
        gameObject.GetComponent<CanvasRenderer>().SetTexture(theTex);
    }

    void DisplayMat(Mat theMat)
    {
        //FOr some reason you have to set up a new fresh mat to copy to when using the Utils.matToTexture2D or it flips it upside down every other time
        Mat tempMat = theMat.clone();
        //theMat.copyTo(tempMat);
        //        Texture2D texturego = new Texture2D(photo1TargetTexture.width, photo1TargetTexture.height);
        Texture2D texturego = new Texture2D(theMat.cols(), theMat.rows());

        Utils.matToTexture2D(tempMat, texturego);

        texturego.Apply(); //OMGFG IT JUST NEEDED APPLY!!! WTF
       // texturego.Resize(displayTexture.width, displayTexture.height);
       //     Graphics.CopyTexture(texturego, displayTexture);
       
       // gameObject.GetComponent<CanvasRenderer>().SetTexture(displayTexture); // Proven works, but flickers off
        //gameObject.GetComponent<CanvasRenderer>().SetTexture(texturego);
         GetComponent<RawImage>().material.mainTexture = displayTexture;


        /*
         * This shit here was causing problems in the regular code
         *     // Texture2D texturego = new Texture2D(photo1TargetTexture.width, photo1TargetTexture.height);

            // Utils.matToTexture2D(photosCam1WB[1], texturego);
            // texturego.Apply();
            //  gameObject.GetComponent<CanvasRenderer>().SetTexture(texturego);
         * 
         */

    }

    void GrabGrayCodePhoto(int index)
    {
        if (pattnum == -1)
        {
            Cam1FinishedProcessing = true;

        }
        else
        {    // string filename = string.Format(@"CapturedImage{0}.png", capturedImageCount);
             //string filePath = System.IO.Path.Combine(Application.persistentDataPath, filename);
            string filePath = "Graycode/Cam1_" +
                index + ".png";
            /*Grab Photo after each Projection*/
            //Save the Photos to the Disk
            //photo1CaptureObject.TakePhotoAsync(filePath, PhotoCaptureFileOutputFormat.PNG, onCapturedPhotoToDiskCam1);
            //Or save the photos to memory
            photo1CaptureObject.TakePhotoAsync(OnCapturedPhotoToMemoryCam1);

        }

    }

    void GrabWBPhoto(int index, string thecolor)
    {
        string filePath = "Graycode/Cam1_" +
           thecolor + ".png";
        /*Grab Photo after each Projection*/
        //Save the Photos to the Disk
        //photo1CaptureObject.TakePhotoAsync(filePath, PhotoCaptureFileOutputFormat.PNG, onCapturedPhotoToDiskCam1);
        //Or save the photos to memory
        photo1CaptureObject.TakePhotoAsync(OnCapturedWBPhotoToMemoryCam1);
    }

    // Computes the shadows occlusion where we cannot reconstruct the model
    void ComputeShadowMasks()
    {
        Debug.Log("~~~Computing Shadow Masks... ");

        Mat whiteMask = new Mat();
        Mat blackMask = new Mat();



        int cam_cols = photosCam1WB[0].cols(); // cam width
        int cam_rows = photosCam1WB[0].rows(); // cam height
                                               //ShadowMask = new Mat(cam_height, cam_width, CvType.CV_8UC1);
                                               //Can we just subtract?
                                               // Core.absdiff(photosCam1WB[0], photosCam1WB[1], ShadowMask);
                                               //ShadowMask = photosCam1WB[0] - photosCam1WB[1];

        //Imgproc.threshold(photosCam1WB[0], ShadowMask, 50, 255, Imgproc.THRESH_BINARY);
        //displayMat(ShadowMask);
        // ShadowMask[0, 1;

        //Make everything gray
        Imgproc.cvtColor(photosCam1WB[0].clone(), whiteMask, Imgproc.COLOR_BGRA2GRAY);
        Imgproc.cvtColor(photosCam1WB[1].clone(), blackMask, Imgproc.COLOR_BGRA2GRAY);
        //photosCam1WB[0].copyTo(ShadowMask);
        ShadowMask = whiteMask.clone();
        ShadowMask.setTo(Scalar.all(100));
        // displayMat(photosCam1WB[0]);
        //Compute the shadowmask 
        //*
        for (int i = 0; i < cam_cols; i++)
        {
            for (int j = 0; j < cam_rows; j++)
            {
                double[] white = whiteMask.get(j, i);



                double[] black = blackMask.get(j, i);

                if (Mathf.Abs((float)white[0] - (float)black[0]) > blackThreshold)
                {
                    byte[] p = new byte[1];
                    p[0] = 255;
                    ShadowMask.put(j, i, p);
                    //.at<uchar>(Point(i, j) ) = (uchar ) 1;
                }
                else
                {
                    byte[] p = new byte[1];
                    p[0] = 0;
                    ShadowMask.put(j, i, p);
                }
            }
        }
        /**/
        DisplayMat(ShadowMask);



    }

    void ReadGrayCodeImages()
    {
        Mat processMatRGB = new Mat();

        List<Mat> photosCam1GRAY = new List<Mat>();

        //photosCam1[0].copyTo(processMatRGB);
        //photosCam1[0].copyTo(ProjPixMat);
        ProjPixMat = photosCam1[0].clone();
        //Get loop of all images
        for (int z = 0; z < photosCam1.Count; z++) // This is all images, should i do horizonatal and vertical separate?
        {
            //Make ALL the captured photos gray
            Debug.Log("~~~Making Images Grayscale... ");
            Imgproc.cvtColor(photosCam1[z], processMatRGB, Imgproc.COLOR_BGRA2GRAY);
            photosCam1GRAY.Add(processMatRGB.clone());


            //Skip pixels outside of shadowmask
            // Mat processMat = new Mat();
            // photosCam1[z].copyTo(processMat, ShadowMask);

        }
        int cam_cols = photosCam1WB[0].cols(); // cam width
        int cam_rows = photosCam1WB[0].rows(); // cam height

        Point projPixel = new Point(0, 0);
       // Mat processMatgrayscale = new Mat();

      //  photosCam1GRAY[0].copyTo(processMatgrayscale);
        //ShadowMask.copyTo(processMat);
      //  processMatgrayscale.setTo(Scalar.all(100), ShadowMask.clone());
        // processMatgrayscale.setTo(Scalar.Equals(0,255,0), ShadowMask);

        //Loop through the standard image size
        /**/
        for (int i = 0; i < cam_cols; i++)
        {
            for (int j = 0; j < cam_rows; j++)
            {
                double[] color1 = { 0, 0, 255, 100 };
                ProjPixMat.put(j, i, color1);

                //if the pixel is not shadowed, reconstruct
                if (ShadowMask.get(j, i)[0] > 0)
                {

                    /**/ //projPixel style
                    //for a (x,y) pixel of the camera returns the corresponding projector pixel by calculating the decimal number
                    bool error = grayCode.getProjPixel(photosCam1GRAY, i, j, projPixel);


                    if (error)
                    {
                        //Debug.Log("~~~Projpix error... ");
                        //Error (can't figure out what the pixel matches to, is Red
                        double[] colorE = { 255, 0, 0, 255 };
                        ProjPixMat.put(j, i, colorE);
                        continue;
                    }
                    /**/
                    /*   END PRODUCT
                    Spit out image grayscale
                    CSV - 4 cols, Xc, Yc, Xp, Yp,
                    Rows - pixels of camera
                    */

                    double[] color = { 0, projPixel.y / projectorPixHeight * 255, projPixel.x / projectorPixWidth * 255, 255 };
                    ProjPixMat.put(j, i, color);
                    Xc.Add(i);
                    Yc.Add(j);
                    Xp.Add((int)projPixel.x);
                    Yp.Add((int)projPixel.y);
                                     
                }
                else // outside the shadow mask WE DONT CARE ABOUT THE PIXEL
                {
                    //Dont care is Blue
                    double[] color = { 0, 0, 255, 255 };
                    ProjPixMat.put(j, i, color);
                }

            }

        }


        Debug.Log("~~~Showing processMat of Test... ");
        
        SaveCSV();
        Debug.Log("~~~Saving Pixel data to CSV... ");

    
        /**/
        DisplayMat(ProjPixMat);



        //Threshold all the images calculated from the W and B photos


        //Create Image Stack

        //Search for pixels


        //Save Pixels with their Gray Code Coordinates

        //Display image with Graycode coordinates mapped to color



        //load cam1 intrinsics and cam 2 intrinsics


        //stereo rectify both images

        // photosCam1






    }

    void SaveCSV()
    {
        string csvfilePath = getPath();

        //This is the writer, it writes to the filepath
        StreamWriter writer = new StreamWriter(csvfilePath);

        //This is writing the line of the type, name, damage... etc... (I set these)
        writer.WriteLine("Xc,Yc,Xp,Yp");
        //This loops through everything in the inventory and sets the file to these.
        for (int i = 0; i < Xc.Count; ++i)
        {
            writer.WriteLine(Xc[i] +
                "," + Yc[i] +
                "," + Xp[i] +
                "," + Yp[i]);
        }
        writer.Flush();
        //This closes the file
        writer.Close();
    }
    private string getPath()
    {
#if UNITY_EDITOR
        // return Application.dataPath + "/Graycode/" + "ProjPoints.csv";

       return "Graycode/"+ "ProjPoints.csv";
#elif UNITY_ANDROID
        return Application.persistentDataPath+"Saved_Inventory.csv";
#elif UNITY_IPHONE
        return Application.persistentDataPath+"/"+"Saved_Inventory.csv";
#else
        return Application.dataPath +"/"+"Saved_Inventory.csv";
#endif
    }
}
