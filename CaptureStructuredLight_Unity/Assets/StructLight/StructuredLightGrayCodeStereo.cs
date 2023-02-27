

using System.Collections.Generic;
using UnityEngine;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityUtils;
using OpenCVForUnity.Structured_lightModule;
using OpenCVForUnity.UtilsModule;

using UnityEngine.Windows.WebCam;

using System.Linq;
using System.IO;
using UnityEngine.UI;
using System.Collections;


public class StructuredLightGrayCodeStereo : MonoBehaviour
{
    Params paramz;
    GrayCodePattern grayCode;

    

    List<Mat> pattern;
    List<Mat> patternWB;
    List<Mat> photosCamA;
    List<Mat> photosCamB;
    List<Mat> photosCamA_WB;//and all white and all black photos
    List<Mat> photosCamB_WB;//and all white and all black photos

    string CURRENTMODE = "PREVIEW";

    WebCamDevice[] devices;

    public bool saveAllGrayCodes = false;
    Mat white;
    Mat black;

    int blackThreshold = 70;  // 3D_underworld default value is 40
    int whiteThreshold = 50;   // 3D_underworld default value is 5

    /// <summary>
    /// The texture.
    /// </summary>
    Texture2D grayCodeTexture;
    Texture2D displayTexture;

    float t = 0f;
    float waittime = 1f;


    int pattnum = -1;

    List<int> XcA, YcA, XcB, YcB, XpA, YpA, XpB, YpB;

    /// <summary>
    /// Set the width projector for scanning.
    /// </summary>
    [SerializeField, TooltipAttribute("Set the width projector for scanning.")]
    public int projectorPixWidth = 1920;

    /// <summary>
    /// Set the height of projector for scanning
    /// </summary>
    [SerializeField, TooltipAttribute("Set the height of projector for scanning.")]
    public int projectorPixHeight = 1080;



    int numOfColImgs;
    int numOfRowImgs;


    //Webcam stuff
    [SerializeField, TooltipAttribute("Choose which camera to use for Cam A.")]
    public int camA_port = 0;
    [SerializeField, TooltipAttribute("Choose which camera to use for Cam B.")]
    public int camB_port = 2;

    ///// <summary>
    ///// Set the name of the device to use.
    ///// </summary>
    //[SerializeField, TooltipAttribute("Set the name of the device to use.")]
    //public string requestedDeviceName = null;lo

    /// <summary>
    /// Set FPS of WebCamTexture.
    /// </summary>
    [SerializeField, TooltipAttribute("Set FPS of WebCamTexture.")]
    public int requestedFPS = 30;

    ///// <summary>
    ///// Set whether to use the front facing camera.
    ///// </summary>
    //[SerializeField, TooltipAttribute("Set whether to use the front facing camera.")]
    //public bool requestedIsFrontFacing = false;

    /// <summary>
    /// The webcam texture.
    /// </summary>
    WebCamTexture webCamTexture;

    /// <summary>
    /// The webcam device.
    /// </summary>
    WebCamDevice webCamDevice;

    bool CamAFinishedProcessing = false;
    bool CamBFinishedProcessing = false;

    bool AllWBcaptured = false;
    bool AllGraycodeCaptured = false;
    bool firstframe = true;
    bool introPreview = true;
    public bool autoCapture = false;


    PhotoCapture photoACaptureObject = null;
    Texture2D photoATargetTexture = null;
    PhotoCapture photoBCaptureObject = null;
    Texture2D photoBTargetTexture = null;
    // Start is called before the first frame update
    WebCamTexture camAwebcam;
    WebCamTexture camBwebcam;

    WebCam webcamA;

    bool cameraArmed = false;


    Mat ShadowMaskA;
    Mat ShadowMaskB;
    Mat ProjPixMatA;
    Mat ProjPixMatB;

    GameObject camAdisplay;
    GameObject camBdisplay;
    GameObject backgroundStripes;

    //Media Device Stuff
    [Header("Camera Preview")]
    public RawImage previewPanel;
    public AspectRatioFitter previewAspectFitter;

    [Header("Photo Capture")]
    public RawImage photoPanel;
    public AspectRatioFitter photoAspectFitter;
    public Image flashIcon;
    public Image switchIcon;
    private bool triggerNewImage = true;
    public bool setCameraSettings = true;


    //timer stuff
    bool timerReached = false;
    double timer = 0;

    void Start()
    {


        camAdisplay = GameObject.Find("CamAImage");
        camBdisplay = GameObject.Find("CamBImage");
        backgroundStripes = GameObject.Find("Background_Stripes");


        //Screen.SetResolution(projectorPixWidth, projectorPixHeight, true);

        pattern = new List<Mat>();
        patternWB = new List<Mat>();
        photosCamA = new List<Mat>();
        photosCamB = new List<Mat>();
        photosCamA_WB = new List<Mat>();
        photosCamB_WB = new List<Mat>();

        //coordinates of projected pixels
        //Xc is camera pixel for cams A and B, Xp is projector location
        XcA = new List<int>();
        YcA = new List<int>();
        XcB = new List<int>();
        YcB = new List<int>();
        XpA = new List<int>();
        YpA = new List<int>();
        XpB = new List<int>();
        YpB = new List<int>();

        black = new Mat();
        white = new Mat();
        ShadowMaskA = new Mat();
        ProjPixMatA = new Mat();
        ShadowMaskB = new Mat();
        ProjPixMatB = new Mat();

        //  paramz = new Params();

        Debug.Log("Projector Size W = " + projectorPixWidth + " H  = " + projectorPixHeight);

        grayCode = GrayCodePattern.create(projectorPixWidth, projectorPixHeight);

        grayCode.generate(pattern);

        Debug.Log("pattern size (OCV) = " + pattern.Count);
        //HMM there is no function to get num of rows vs columns grayCode.get
        grayCode.getImagesForShadowMasks(black, white);
        patternWB.Add(white);
        patternWB.Add(black);

        Debug.Log("patternWB size = " + patternWB.Count);

        SaveAllGrayCodetoPNG();
        InitializeCameras();

    }


    // Update is called once per frame
    void Update()
    {

        //Set Cursor to not be visible
        Cursor.visible = false;
        //Cursor.lockState = CursorLockMode.Locked;

        switch (CURRENTMODE)
        {
            case "PREVIEW":
                //Show the initial camera video for focus and adjustmenta
                //gameObject.GetComponentInChildren<RawImage>().material.mainTexture = photoATargetTexture;
                camAdisplay.GetComponent<Image>().material.mainTexture = camAwebcam;
                camBdisplay.GetComponent<Image>().material.mainTexture = camBwebcam;
                checkKeys();

                if (Input.GetKeyDown(KeyCode.A) && !autoCapture)
                {
                    CURRENTMODE = "CAPWB";
                    camAdisplay.SetActive(false);
                    camBdisplay.SetActive(false);
                    backgroundStripes.SetActive(false);

                    autoCapture = true;
                    Debug.Log("AUTOCAPTURE ON");


                }
                //Swithc to capturing White and Black photos
                if (Input.GetKeyDown(KeyCode.Space))
                {
                    CURRENTMODE = "CAPWB";
                    camAdisplay.SetActive(false);
                    camBdisplay.SetActive(false);
                    backgroundStripes.SetActive(false);

                    Debug.Log("GOING TO CAPTURE WB MODE");

                }
                break;

            case "CAPWB":

                //Take photo of graycode
                if (cameraArmed)
                {
                    if (Input.GetKeyDown(KeyCode.Space) || (autoCapture == true))
                    {
                        cameraArmed = false;

                        //timer += Time.realtimeSinceStartup;
                        //double thedelay = 2.5 + timer;
                        //while (timer < thedelay)
                        //{
                        //    timer = Time.realtimeSinceStartup;
                        //    //Debug.Log(" WB waiting for secs..." + (thedelay - timer));
                        //}
                        //timer = 0;
                        Debug.Log("Done waiting WB photo");
                        if (pattnum == -1)
                        { triggerNewImage = true; }
                        if (pattnum == 0)
                        {
                            GrabWBPhotos(pattnum, "white");
                        }
                        if (pattnum == 1) GrabWBPhotos(pattnum, "black");
                    }

                }
                //Display the All White and All Black images for the mask
                else
                {
                    if (triggerNewImage)
                    {
                        pattnum++;
                        if (pattnum > patternWB.Count - 1)
                        {
                            AllWBcaptured = true;
                            pattnum = -1;
                            CURRENTMODE = "CAPGRAY";
                            //CURRENTMODE = "PROCESS";
                            triggerNewImage = true;
                            Debug.Log("GOING TO CAPTURE GRAYCODES MODE");

                        }
                        else
                        {
                            if (pattnum > -1)
                            {
                                DisplayMat(patternWB[pattnum]);
                            }
                            Debug.Log("Showing WB Pattern: " + pattnum);
                            triggerNewImage = false;

                            StartCoroutine(waitUntilCameraArmed());
                            //timer += Time.realtimeSinceStartup;

                            //double thedelay = 2.5 + timer;
                            //while (timer < thedelay)
                            //{
                            //    timer = Time.realtimeSinceStartup;
                            //    //Debug.Log(" WB waiting for secs..." + (thedelay - timer));
                            //}
                            //timer = 0;

                        }
                    }

                }

                checkKeys();

                break;

            case "CAPGRAY":
                // Display a gray code series and then capture them


                //Take photo of graycode
                if (cameraArmed)
                {
                    if (Input.GetKeyDown(KeyCode.Space) || (autoCapture == true))   
                    {
                        cameraArmed = false;

                        //timer += Time.realtimeSinceStartup;
                        //double thedelay = .4 + timer;
                        //while (timer < thedelay)
                        //{
                        //    timer = Time.realtimeSinceStartup;
                        //    // Debug.Log(" graycode waiting for secs..." + (thedelay- timer));
                        //}
                        //timer = 0;
                        // Debug.Log("Done waiting graycode photo");
                        GrabGrayCodePhotos(pattnum);

                    }

                }

                //Display the All White and All Black images for the mask
                else
                {
                    if (triggerNewImage)
                    {
                        pattnum++;
                        if (pattnum > pattern.Count - 1)
                        {
                            AllGraycodeCaptured = true;
                            pattnum = -1;
                            CURRENTMODE = "PROCESS";
                            triggerNewImage = true;
                            camAwebcam.Stop();
                            camBwebcam.Stop();
                            autoCapture = false;

                            camAdisplay.SetActive(true);
                            camBdisplay.SetActive(true);
                            Debug.Log("Webcam Stopped");

                            Debug.Log("Finished capturing all images, press P to process, Press S to save all cap img to disk");
                        }
                        else
                        {
                            DisplayMat(pattern[pattnum]);
                            Debug.Log("Currently Showing Graycode Pattern: " + pattnum + " / " + pattern.Count);
                            triggerNewImage = false;
                            StartCoroutine(waitUntilCameraArmed());


                        }
                    }
                }

                checkKeys();


                break;


            case "PROCESS":

                if (Input.GetKeyDown(KeyCode.S))
                {
                    Debug.Log("Saving photos... ");

                    SaveAllCaptures();
                }
                //Process Images
                if (Input.GetKeyDown(KeyCode.P))
                {
                    introPreview = false;

                    Debug.Log("Processing the Graycode... ");

                    ComputeShadowMasks();
                    ReadGrayCodeImages();

                }


                break;
            default:
                //some code 
                Debug.LogError("this shouldn't be happening default case ");

                break;

        }

    }

    IEnumerator waitUntilCameraArmed()
    {
         yield return new WaitForSecondsRealtime(0.65f);
        //yield return new WaitForEndOfFrame();// WaitForSecondsRealtime(.55f); //doens't work
       // yield return new wait


        Debug.Log("Camera Armed");

        cameraArmed = true;
    }

    void checkKeys()
    {
        if (Input.GetKeyDown(KeyCode.S))
        {
            Debug.Log("Start Saving all pics");

    

            SaveAllCaptures();
        }
        if (Input.GetKeyDown(KeyCode.L))
        {
            LoadPreviousCaptures();
            CURRENTMODE = "PROCESS";
        }
        if (Input.GetKeyDown(KeyCode.A) && !autoCapture)
        {
            autoCapture = true;
            Debug.Log("AUTOCAPTURE ON");
        }


    }

    void LoadPreviousCaptures()
    {
        introPreview = false;
        //Load a photo for every graycode generated
        Debug.Log("Loading Previous Graycode Camera Captures");
        //Stop the webcams if they are running
        camAwebcam.Stop();
        camBwebcam.Stop();
        autoCapture = false;
        camAdisplay.SetActive(false);
        camBdisplay.SetActive(false);
        Debug.Log("Webcam Stopped");

        //first clear all any photos that may have been stored
        photosCamA.Clear();
        photosCamA_WB.Clear();
        //first clear all any photos that may have been stored
        photosCamB.Clear();
        photosCamB_WB.Clear();

        for (int i = 0; i < grayCode.getNumberOfPatternImages(); i++)
        {
            // Create an array of file paths from which to choose
            //Below is an example to read all PNG's from a folder
            //folderPath = Application.streamingAssetsPath + "/NameImages/GameNames/";  //Get path of folder
            //       filePaths = Directory.GetFiles(folderPath, "*.png"); // Get all files of type .png in this folder

            string filePathA = "Graycode/CamA_" +
              i + ".png";
            string filePathB = "Graycode/CamB_" +
                i + ".png";
            //Converts desired path into byte array
            byte[] pngBytesA = File.ReadAllBytes(filePathA);
            byte[] pngBytesB = File.ReadAllBytes(filePathB);

            //Creates texture and loads byte array data to create image
            Texture2D texA = new Texture2D(photoATargetTexture.width, photoATargetTexture.height);
            Texture2D texB = new Texture2D(photoBTargetTexture.width, photoBTargetTexture.height);

            texA.LoadImage(pngBytesA);
            texB.LoadImage(pngBytesB);

            //TODO make these mats reconfigure to the size of the graycode detected
            Mat matA = new Mat(photoATargetTexture.height, photoATargetTexture.width, CvType.CV_8UC4);
            Mat matB = new Mat(photoBTargetTexture.height, photoBTargetTexture.width, CvType.CV_8UC4);

            Utils.texture2DToMat(texA, matA);
            photosCamA.Add(matA);
            Utils.texture2DToMat(texB, matB);
            photosCamB.Add(matB);


        }
        //Get the white and black images
        for (int i = 0; i < 2; i++)
        {
            string filePathA = "Graycode/CamA_WB_" +
              i + ".png";
            string filePathB = "Graycode/CamB_WB_" +
          i + ".png";
            //Converts desired path into byte array
            byte[] pngBytesA = File.ReadAllBytes(filePathA);
            byte[] pngBytesB = File.ReadAllBytes(filePathB);

            //Creates texture and loads byte array data to create image
            Texture2D texA = new Texture2D(photoATargetTexture.width, photoATargetTexture.height);
            Texture2D texB = new Texture2D(photoBTargetTexture.width, photoBTargetTexture.height);

            texA.LoadImage(pngBytesA);
            texB.LoadImage(pngBytesB);

            Mat matA = new Mat(photoATargetTexture.height, photoATargetTexture.width, CvType.CV_8UC4);
            Mat matB = new Mat(photoBTargetTexture.height, photoBTargetTexture.width, CvType.CV_8UC4);

            Utils.texture2DToMat(texA, matA);
            Utils.texture2DToMat(texB, matB);

            photosCamA_WB.Add(matA);
            photosCamB_WB.Add(matB);
        }

        Debug.Log("Loaded Cam A Graycode Captures total = " + photosCamA.Count + " and total CamA WB captures = " + photosCamA_WB.Count);
        Debug.Log("Loaded Graycode B Captures Total Graycodes = " + photosCamB.Count + " and total CamB WB captures = " + photosCamB_WB.Count);

        //Display one of the captures

        camAdisplay.SetActive(true);
        camBdisplay.SetActive(true);
        //DisplayMat(photosCamA_WB[0]);
        DisplayMatCamA(photosCamA_WB[0]);
        DisplayMatCamB(photosCamB_WB[1]);
    }


    void SaveAllCaptures()
    {
        introPreview = false;

        Debug.Log("Saving ALL Camera imgs... ");

        //Save all Captured Textures to Disk
        for (int i = 0; i < photosCamA.Count; i++)
        {
            Texture2D theTex = new Texture2D(photoATargetTexture.width, photoATargetTexture.height);
            // Debug.Log("photosCamA[i] width " + photosCamA[i].width());

            //Save image to disk
            Utils.matToTexture2D(photosCamA[i], theTex);

            byte[] bytes = theTex.EncodeToPNG();
            string filePath = "Graycode/CamA_" +
               i + ".png";
            File.WriteAllBytes(Application.absoluteURL + filePath, bytes);
            // Debug.Log("Saving Camera img PNG: " + filePath);
        }
        for (int i = 0; i < photosCamA_WB.Count; i++)
        {
            Texture2D theTex = new Texture2D(photoATargetTexture.width, photoATargetTexture.height);

            //Save image to disk
            Utils.matToTexture2D(photosCamA_WB[i], theTex);

            byte[] bytes = theTex.EncodeToPNG();
            string filePath = "Graycode/CamA_WB_" +
               i + ".png";
            File.WriteAllBytes(Application.absoluteURL + filePath, bytes);
            // Debug.Log("Saving Camera img PNG: " + filePath);
        }
        Debug.Log("CamA Saved all graycode and WB captures ");

        //CAM B Save all Captured Textures to Disk
        for (int i = 0; i < photosCamB.Count; i++)
        {
            Texture2D theTex = new Texture2D(photoBTargetTexture.width, photoBTargetTexture.height);

            //Save image to disk
            Utils.matToTexture2D(photosCamB[i], theTex);

            byte[] bytes = theTex.EncodeToPNG();
            string filePath = "Graycode/CamB_" +
               i + ".png";
            File.WriteAllBytes(Application.absoluteURL + filePath, bytes);
            // Debug.Log("Saving Camera img PNG: " + filePath);
        }
        for (int i = 0; i < photosCamB_WB.Count; i++)
        {
            Texture2D theTex = new Texture2D(photoBTargetTexture.width, photoBTargetTexture.height);

            //Save image to disk
            Utils.matToTexture2D(photosCamB_WB[i], theTex);

            byte[] bytes = theTex.EncodeToPNG();
            string filePath = "Graycode/CamB_WB_" +
               i + ".png";
            File.WriteAllBytes(Application.absoluteURL + filePath, bytes);
            //Debug.Log("Saving Camera img PNG: " + filePath);
        }
        Debug.Log("CamB Saved all graycode and WB captures ");

    }

    void SaveMattoPNG(Mat themat, string nameofimage)
    {
        Texture2D theTex = new Texture2D(themat.width(), themat.height());

        //Save image to disk
        Utils.matToTexture2D(themat.clone(), theTex);

        byte[] bytes = theTex.EncodeToPNG();
        string filePath = "Graycode/" + nameofimage + ".png";
        File.WriteAllBytes(Application.absoluteURL + filePath, bytes);
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
                // Debug.Log("Saving Graycode PNG: " + filePath);
            }
        }
        Debug.Log("Saved all graycode Files");
    }



    void InitializeCameras()
    {
        //Setup Webcams
        devices = WebCamTexture.devices;
        for (int i = 0; i < devices.Length; i++) // How many webcams we got?
            Debug.Log("Webcam " + i + " name = " + devices[i].name);

        Debug.Log("Photocapture Camera resolution" + PhotoCapture.SupportedResolutions.OrderByDescending((res) => res.width * res.height).First()); //get highest resolution camera
        List<Resolution> reses = PhotoCapture.SupportedResolutions.ToList();

        for (int i = 0; i < reses.Count; i++)
        {
            Debug.Log($"width {reses[i].width} and height {reses[i].height}");

        }

        Resolution cameraResolution = PhotoCapture.SupportedResolutions.OrderByDescending((res) => res.width * res.height).First();
        photoATargetTexture = new Texture2D(cameraResolution.width, cameraResolution.height);
        photoBTargetTexture = new Texture2D(cameraResolution.width, cameraResolution.height);
        displayTexture = new Texture2D(photoATargetTexture.width, photoATargetTexture.height);
        //devices[0].

        if (setCameraSettings)
        {
            //Pop up dialog for camera control!
            OpenCVForUnity.VideoioModule.VideoCapture vidcap0;
            vidcap0 = new OpenCVForUnity.VideoioModule.VideoCapture();
            vidcap0.open(camA_port, OpenCVForUnity.VideoioModule.Videoio.CAP_DSHOW);

            vidcap0.set(OpenCVForUnity.VideoioModule.Videoio.CAP_PROP_SETTINGS, 0);

            vidcap0.release();

            vidcap0.open(camB_port, OpenCVForUnity.VideoioModule.Videoio.CAP_DSHOW);
            vidcap0.set(OpenCVForUnity.VideoioModule.Videoio.CAP_PROP_SETTINGS, 0);
            vidcap0.release();


        }




        camAwebcam = new WebCamTexture(devices[camA_port].name, cameraResolution.width, cameraResolution.height, 60);
       // Debug.Log("Cam A FPS " + camAwebcam.requestedFPS);
        camBwebcam = new WebCamTexture(devices[camB_port].name, cameraResolution.width, cameraResolution.height, 60);
       // Debug.Log("Cam B FPS " + camBwebcam.requestedFPS);

        camAwebcam.Play();
        camBwebcam.Play();


        /**
        // Create a PhotoCapture object for CamA
        PhotoCapture.CreateAsync(false, delegate (PhotoCapture captureObjectA)
        {
            photoACaptureObject = captureObjectA;

            CameraParameters cameraParameters = new CameraParameters();
            cameraParameters.hologramOpacity = 0.0f;
            cameraParameters.cameraResolutionWidth = cameraResolution.width;
            cameraParameters.cameraResolutionHeight = cameraResolution.height;
            cameraParameters.pixelFormat = CapturePixelFormat.BGRA32;

                // Activate the camera
                photoACaptureObject.StartPhotoModeAsync(cameraParameters, delegate (PhotoCapture.PhotoCaptureResult result)
            {
                    // Take an initial picture
                    //photo1CaptureObject.TakePhotoAsync(OnCapturedPhotoToMemoryCam1);
                });
        });
        // Create a PhotoCapture object for CamB
        PhotoCapture.CreateAsync(false, delegate (PhotoCapture captureObjectB)
        {
            photoBCaptureObject = captureObjectB;

            CameraParameters cameraParameters = new CameraParameters();
            cameraParameters.hologramOpacity = 0.0f;
            cameraParameters.cameraResolutionWidth = cameraResolution.width;
            cameraParameters.cameraResolutionHeight = cameraResolution.height;
            cameraParameters.pixelFormat = CapturePixelFormat.BGRA32;

                // Activate the camera
                photoBCaptureObject.StartPhotoModeAsync(cameraParameters, delegate (PhotoCapture.PhotoCaptureResult resultB)
            {
                    // Take an initial picture
                    //photo1CaptureObject.TakePhotoAsync(OnCapturedPhotoToMemoryCam1);
                });
        });
        /**/

    }


    void TakeSnapshotCamA()
    {
       // if (!camAwebcam.didUpdateThisFrame)
        {
            photoATargetTexture = new Texture2D(camAwebcam.width, camAwebcam.height);
            photoATargetTexture.SetPixels(camAwebcam.GetPixels());
            photoATargetTexture.Apply();

            /* System.IO.File.WriteAllBytes(_SavePath + _CaptureCounter.ToString() + ".png", snap.EncodeToPNG());
             ++_CaptureCounter;
             */

            Debug.Log("camA to TexA ~~~  Captured photo at index: " + pattnum);
            Mat camAMat = new Mat(photoATargetTexture.height, photoATargetTexture.width, CvType.CV_8UC4);

            Utils.texture2DToMat(photoATargetTexture, camAMat);
            photosCamA.Add(camAMat);
            Debug.Log("MatPhoto from camA added to photoscamA at index: " + (photosCamA.Count - 1));

            CamAFinishedProcessing = true;
        }
    }
    void TakeSnapshotCamB()
    {
       // if (camBwebcam.didUpdateThisFrame)
        {
            photoBTargetTexture = new Texture2D(camBwebcam.width, camBwebcam.height);
            photoBTargetTexture.SetPixels(camBwebcam.GetPixels());
            photoBTargetTexture.Apply();

            /* System.IO.File.WriteAllBytes(_SavePath + _CaptureCounter.ToString() + ".png", snap.EncodeToPNG());
             ++_CaptureCounter;
             */
            Debug.Log("camB to TexB ~~~  Captured photo at index: " + pattnum);
            Mat camBMat = new Mat(photoBTargetTexture.height, photoBTargetTexture.width, CvType.CV_8UC4);

            Utils.texture2DToMat(photoBTargetTexture, camBMat);
            photosCamB.Add(camBMat);
            Debug.Log("MatPhoto from camB added to photoscamB at index: " + (photosCamB.Count - 1));

            CamBFinishedProcessing = true;
        }
    }

    void TakeSnapshotCamA_WB()
    {

        
        {
            photoATargetTexture = new Texture2D(camAwebcam.width, camAwebcam.height);
           
            photoATargetTexture.SetPixels(camAwebcam.GetPixels());
            photoATargetTexture.Apply();

            /* System.IO.File.WriteAllBytes(_SavePath + _CaptureCounter.ToString() + ".png", snap.EncodeToPNG());
             ++_CaptureCounter;
             */

            Debug.Log("camA to TexA ~~~  Captured photo at index: " + pattnum);
            Mat camAMat = new Mat(photoATargetTexture.height, photoATargetTexture.width, CvType.CV_8UC4);


            Utils.texture2DToMat(photoATargetTexture, camAMat);

            photosCamA_WB.Add(camAMat);
            Debug.Log("MatPhoto from camA added to photoscamA_WB at index: " + (photosCamA_WB.Count - 1));

            CamAFinishedProcessing = true;
        }
    }

    void TakeSnapshotCamB_WB()


    {

       // if (camBwebcam.didUpdateThisFrame)
        {
            photoBTargetTexture = new Texture2D(camBwebcam.width, camBwebcam.height);
            photoBTargetTexture.SetPixels(camBwebcam.GetPixels());
            photoBTargetTexture.Apply();

            /* System.IO.File.WriteAllBytes(_SavePath + _CaptureCounter.ToString() + ".png", snap.EncodeToPNG());
             ++_CaptureCounter;
             */
            Debug.Log("camB to TexB ~~~  Captured photo at index: " + pattnum);
            Mat camBMat = new Mat(photoBTargetTexture.height, photoBTargetTexture.width, CvType.CV_8UC4);


            Utils.texture2DToMat(photoBTargetTexture, camBMat);

            photosCamB_WB.Add(camBMat);
            Debug.Log("MatPhoto from camB added to photoscamB_WB at index: " + (photosCamB_WB.Count - 1));

            CamBFinishedProcessing = true;
        }

    }


    void OnCapturedPhotoToMemoryCamA(PhotoCapture.PhotoCaptureResult result, PhotoCaptureFrame photoCaptureFrame)
    {

        // Copy the raw image data into our target texture
        photoCaptureFrame.UploadImageDataToTexture(photoATargetTexture);
        Debug.Log("camA to TexA ~~~  Captured photo at index: " + pattnum);
        Mat camAMat = new Mat(photoATargetTexture.height, photoATargetTexture.width, CvType.CV_8UC4);

        Utils.texture2DToMat(photoATargetTexture, camAMat);
        photosCamA.Add(camAMat);
        Debug.Log("MatPhoto from camA added to photoscamA at index: " + (photosCamA.Count - 1));

        CamAFinishedProcessing = true;
    }

    void OnCapturedPhotoToMemoryCamB(PhotoCapture.PhotoCaptureResult result, PhotoCaptureFrame photoCaptureFrame)
    {
        // Copy the raw image data into our target texture
        photoCaptureFrame.UploadImageDataToTexture(photoBTargetTexture);
        Debug.Log("camB to TexB ~~~  Captured photo at index: " + pattnum);
        Mat camBMat = new Mat(photoBTargetTexture.height, photoBTargetTexture.width, CvType.CV_8UC4);

        Utils.texture2DToMat(photoBTargetTexture, camBMat);
        photosCamB.Add(camBMat);
        Debug.Log("MatPhoto from camB added to photoscamB at index: " + (photosCamB.Count - 1));

        CamBFinishedProcessing = true;
    }


    void OnCapturedWBPhotoToMemoryCamA(PhotoCapture.PhotoCaptureResult result, PhotoCaptureFrame photoCaptureFrame)
    {
        // Copy the raw image data into our target texture
        photoCaptureFrame.UploadImageDataToTexture(photoATargetTexture);
        Debug.Log("camA to TexA ~~~  Captured photo at index: " + pattnum);
        Mat camAMat = new Mat(photoATargetTexture.height, photoATargetTexture.width, CvType.CV_8UC4);


        Utils.texture2DToMat(photoATargetTexture, camAMat);

        photosCamA_WB.Add(camAMat);
        Debug.Log("MatPhoto from camA added to photoscamA_WB at index: " + (photosCamA_WB.Count - 1));

        CamAFinishedProcessing = true;
    }

    void OnCapturedWBPhotoToMemoryCamB(PhotoCapture.PhotoCaptureResult result, PhotoCaptureFrame photoCaptureFrame)
    {
        // Copy the raw image data into our target texture
        photoCaptureFrame.UploadImageDataToTexture(photoBTargetTexture);
        Debug.Log("camB to TexB ~~~  Captured photo at index: " + pattnum);
        Mat camBMat = new Mat(photoBTargetTexture.height, photoBTargetTexture.width, CvType.CV_8UC4);


        Utils.texture2DToMat(photoBTargetTexture, camBMat);

        photosCamB_WB.Add(camBMat);
        Debug.Log("MatPhoto from camB added to photoscamB_WB at index: " + (photosCamB_WB.Count - 1));

        CamAFinishedProcessing = true;
    }

    void OnCapturedPhotoToDiskCamA(PhotoCapture.PhotoCaptureResult result)
    {
        CamAFinishedProcessing = true;
        Debug.Log("Saved CamA Image to disk at index: " + pattnum);
        //Save image to disk
        /* byte[] bytes = photo1TargetTexture.EncodeToPNG();
         string filePath = "Graycode/Cam1_" +
            index + ".png";
         File.WriteAllBytes(Application.absoluteURL + filePath, bytes);
         Debug.Log("Saving Camera img PNG: " + filePath);
         */
    }

    void OnCapturedPhotoToDiskCamB(PhotoCapture.PhotoCaptureResult result)
    {
        CamBFinishedProcessing = true;
        Debug.Log("Saved CamB Image to disk at index: " + pattnum);
        //Save image to disk
        /* byte[] bytes = photo1TargetTexture.EncodeToPNG();
         string filePath = "Graycode/Cam1_" +
            index + ".png";
         File.WriteAllBytes(Application.absoluteURL + filePath, bytes);
         Debug.Log("Saving Camera img PNG: " + filePath);
         */
    }

    void OnStoppedPhotoModeA(PhotoCapture.PhotoCaptureResult result)
    {
        // Shutdown our photo capture resource
        photoACaptureObject.Dispose();
        photoACaptureObject = null;
    }
    void OnStoppedPhotoModeB(PhotoCapture.PhotoCaptureResult result)
    {
        // Shutdown our photo capture resource
        photoBCaptureObject.Dispose();
        photoBCaptureObject = null;
    }
    void DisplayTexture(Texture2D theTex)
    {

        theTex.Apply(); //OMGFG IT JUST NEEDED APPLY!!! WTF
        gameObject.GetComponent<CanvasRenderer>().SetTexture(theTex);
    }

    void DisplayMatCamA(Mat theMat)
    {
        //FOr some reason you have to set up a new fresh mat to copy to when using the Utils.matToTexture2D or it flips it upside down every other time
        Mat tempMat = theMat.clone();
        Texture2D texturego = new Texture2D(theMat.cols(), theMat.rows());
        Utils.matToTexture2D(tempMat, texturego);

        texturego.Apply(); //OMGFG IT JUST NEEDED APPLY!!! WTF
                           // displayTexture.Resize(theMat.cols(), theMat.rows());
                           // displayTexture.Apply();
                           // Graphics.CopyTexture(texturego, displayTexture);
                           // displayTexture.Apply();
        camAdisplay.GetComponent<Image>().material.mainTexture = texturego;


    }

    void DisplayMatCamB(Mat theMat)
    {
        //FOr some reason you have to set up a new fresh mat to copy to when using the Utils.matToTexture2D or it flips it upside down every other time
        Mat tempMat = theMat.clone();
        Texture2D texturego = new Texture2D(theMat.cols(), theMat.rows());
        Utils.matToTexture2D(tempMat, texturego);

        texturego.Apply(); //OMGFG IT JUST NEEDED APPLY!!! WTF
                           //   displayTexture.Resize(theMat.cols(), theMat.rows());
                           //  displayTexture.Apply();
                           //  Graphics.CopyTexture(texturego, displayTexture);
                           // displayTexture.Apply();
        camBdisplay.GetComponent<Image>().material.mainTexture = texturego;


    }


    void DisplayMat(Mat theMat)
    {
        //FOr some reason you have to set up a new fresh mat to copy to when using the Utils.matToTexture2D or it flips it upside down every other time
        Mat tempMat = theMat.clone();
        Texture2D texturego = new Texture2D(theMat.cols(), theMat.rows());
        Utils.matToTexture2D(tempMat, texturego);

        texturego.Apply(); //OMGFG IT JUST NEEDED APPLY!!! WTF
        displayTexture.Resize(theMat.cols(), theMat.rows());
        displayTexture.Apply();
        Graphics.CopyTexture(texturego, displayTexture);
        displayTexture.Apply();
        GetComponent<RawImage>().texture = displayTexture;


    }

  void GrabGrayCodePhotos(int index)
    {
        //yield return new WaitForSecondsRealtime(0.25f);

        CamAFinishedProcessing = false;
        CamBFinishedProcessing = false;

        // string filename = string.Format(@"CapturedImage{0}.png", capturedImageCount);
        //string filePath = System.IO.Path.Combine(Application.persistentDataPath, filename);
        string filePathCamA = "Graycode/CamA_" +
            index + ".png";
        string filePathCamB = "Graycode/CamB_" +
            index + ".png";
        /*Grab Photo after each Projection*/
        //Save the Photos to the Disk
        //photo1CaptureObject.TakePhotoAsync(filePath, PhotoCaptureFileOutputFormat.PNG, onCapturedPhotoToDiskCam1);

        //Or save the photos to memory
        //photoACaptureObject.TakePhotoAsync(OnCapturedPhotoToMemoryCamA);
        //photoBCaptureObject.TakePhotoAsync(OnCapturedPhotoToMemoryCamB);
        TakeSnapshotCamA();
        TakeSnapshotCamB();

        if (CamAFinishedProcessing && CamBFinishedProcessing)
        {
            triggerNewImage = true;
        }

       // yield return new WaitForSecondsRealtime(1.25f);


       // yield return null;
    }

void GrabWBPhotos(int index, string thecolor)
    {
        // yield return new WaitForSeconds(1.25f);
       // yield return new WaitForEndOfFrame();

        CamAFinishedProcessing = false;
        CamBFinishedProcessing = false;

        string filePathA = "Graycode/CamA_" +
           thecolor + ".png";
        string filePathB = "Graycode/CamB_" +
         thecolor + ".png";
        /*Grab Photo after each Projection*/
        //Save the Photos to the Disk
        //photo1CaptureObject.TakePhotoAsync(filePath, PhotoCaptureFileOutputFormat.PNG, onCapturedPhotoToDiskCam1);
        //Or save the photos to memory
        // photoACaptureObject.TakePhotoAsync(OnCapturedWBPhotoToMemoryCamA);
        //photoBCaptureObject.TakePhotoAsync(OnCapturedWBPhotoToMemoryCamB);
        camAwebcam.Pause();
        camBwebcam.Pause();
        camAwebcam.Play();
        camBwebcam.Play();
        TakeSnapshotCamA_WB();
        TakeSnapshotCamB_WB();

        if (CamAFinishedProcessing && CamBFinishedProcessing)
        {
            triggerNewImage = true;
        }
       // yield return new WaitForSeconds(1.25f);

        //yield return null;
    }

    // Computes the shadows occlusion where we cannot reconstruct the model
    void ComputeShadowMasks()
    {
        Debug.Log("~~~Computing Shadow Masks... ");

        Mat whiteMaskA = new Mat();
        Mat blackMaskA = new Mat();
        Mat whiteMaskB = new Mat();
        Mat blackMaskB = new Mat();



        int cam_colsA = photosCamA_WB[0].cols(); // cam width
        int cam_rowsA = photosCamA_WB[0].rows(); // cam height

        int cam_colsB = photosCamB_WB[0].cols(); // cam width
        int cam_rowsB = photosCamB_WB[0].rows(); // cam height
                                                 //ShadowMask = new Mat(cam_height, cam_width, CvType.CV_8UC1);
                                                 //Can we just subtract?
                                                 // Core.absdiff(photosCam1WB[0], photosCam1WB[1], ShadowMask);
                                                 //ShadowMask = photosCam1WB[0] - photosCam1WB[1];

        //Imgproc.threshold(photosCam1WB[0], ShadowMask, 50, 255, Imgproc.THRESH_BINARY);
        //displayMat(ShadowMask);
        // ShadowMask[0, 1;

        //Make everything gray
        Imgproc.cvtColor(photosCamA_WB[0].clone(), whiteMaskA, Imgproc.COLOR_BGRA2GRAY);
        Imgproc.cvtColor(photosCamA_WB[1].clone(), blackMaskA, Imgproc.COLOR_BGRA2GRAY);

        Imgproc.cvtColor(photosCamB_WB[0].clone(), whiteMaskB, Imgproc.COLOR_BGRA2GRAY);
        Imgproc.cvtColor(photosCamB_WB[1].clone(), blackMaskB, Imgproc.COLOR_BGRA2GRAY);

        ShadowMaskA = whiteMaskA.clone();
        ShadowMaskA.setTo(Scalar.all(100));

        ShadowMaskB = whiteMaskB.clone();
        ShadowMaskB.setTo(Scalar.all(100));

        // displayMat(photosCam1WB[0]);
        //Compute the shadowmaskS 
        //*
        for (int i = 0; i < cam_colsA; i++)
        {
            for (int j = 0; j < cam_rowsA; j++)
            {
                double[] white = whiteMaskA.get(j, i);



                double[] black = blackMaskA.get(j, i);

                if (Mathf.Abs((float)white[0] - (float)black[0]) > blackThreshold)
                {
                    byte[] p = new byte[1];
                    p[0] = 255;
                    ShadowMaskA.put(j, i, p);
                    //.at<uchar>(Point(i, j) ) = (uchar ) 1;
                }
                else
                {
                    byte[] p = new byte[1];
                    p[0] = 0;
                    ShadowMaskA.put(j, i, p);
                }
            }
        }
        /**/
        // DisplayMat(ShadowMaskA);
        DisplayMatCamA(ShadowMaskA);

        //Repeat for Shadow b
        for (int i = 0; i < cam_colsB; i++)
        {
            for (int j = 0; j < cam_rowsB; j++)
            {
                double[] white = whiteMaskB.get(j, i);



                double[] black = blackMaskB.get(j, i);

                if (Mathf.Abs((float)white[0] - (float)black[0]) > blackThreshold)
                {
                    byte[] p = new byte[1];
                    p[0] = 255;
                    ShadowMaskB.put(j, i, p);
                    //.at<uchar>(Point(i, j) ) = (uchar ) 1;
                }
                else
                {
                    byte[] p = new byte[1];
                    p[0] = 0;
                    ShadowMaskB.put(j, i, p);
                }
            }
        }
        /**/
        DisplayMatCamB(ShadowMaskB);


    }

    void ReadGrayCodeImages()
    {
        Mat processMatRGB = new Mat();
        Mat processMatRGBB = new Mat();

        List<Mat> photosCamAGRAY = new List<Mat>();
        List<Mat> photosCamBGRAY = new List<Mat>();

        //photosCam1[0].copyTo(processMatRGB);
        //photosCam1[0].copyTo(ProjPixMat);
        ProjPixMatA = photosCamA[0].clone();
        ProjPixMatB = photosCamB[0].clone();

        //Load all images from Cam A and Cam B and make GRAY
        for (int z = 0; z < photosCamA.Count; z++)
        {
            //Make ALL the captured photos gray
            Debug.Log("~~~Making Images Grayscale... ");
            Imgproc.cvtColor(photosCamA[z], processMatRGB, Imgproc.COLOR_BGRA2GRAY);
            photosCamAGRAY.Add(processMatRGB.clone());

            Imgproc.cvtColor(photosCamB[z], processMatRGBB, Imgproc.COLOR_BGRA2GRAY);
            photosCamBGRAY.Add(processMatRGBB.clone());
        }

        int cam_colsA = photosCamA_WB[0].cols(); // cam width
        int cam_rowsA = photosCamA_WB[0].rows(); // cam height
        int cam_colsB = photosCamB_WB[0].cols(); // cam width
        int cam_rowsB = photosCamB_WB[0].rows(); // cam height
        Point projPixelA = new Point(0, 0);
        Point projPixelB = new Point(0, 0);


        //Loop through the images from Cam A
        /**/
        for (int i = 0; i < cam_colsA; i++)
        {
            for (int j = 0; j < cam_rowsA; j++)
            {
                double[] color1 = { 0, 0, 0, 100 };
                ProjPixMatA.put(j, i, color1);

                //if the pixel is not shadowed, reconstruct
                //TODO save shadowMask to files
                //TODO change the Puts of how we are setting the pixels
                // img<uchar>.at(row,col) = 0
                // img.at<Vec3b>(y, x)

                //img.at<Vec3b>(y,x) = intensity
                //Vec4b & bgra = mat.at<Vec4b>(i, j);
                //Vec4b intensity(b, g, r, a)
                //mat.at<Vec4b>(i,j) = intensity
                //find the                 ShadowMaskA.type();



                if (ShadowMaskA.get(j, i)[0] > 0)
                {

                    /**/ //projPixel style
                         //for a (x,y) pixel of the camera returns the corresponding projector pixel by calculating the decimal number
                    bool error = grayCode.getProjPixel(photosCamAGRAY, i, j, projPixelA);


                    if (error)
                    {
                        //Debug.Log("~~~Projpix error... ");
                        //Error (can't figure out what the pixel matches to, is Red
                         double[] colorE = { 255, 0, 0, 40 };
                         ProjPixMatA.put(j, i, colorE);
                        continue;
                    }
                    /**/
                    /*   END PRODUCT
                    Spit out image grayscale
                    CSV - 4 cols, Xc, Yc, Xp, Yp,
                    Rows - pixels of camera
                    */

                    double[] color = { 0, projPixelA.y / projectorPixHeight * 255, projPixelA.x / projectorPixWidth * 255, 255 };
                    ProjPixMatA.put(j, i, color);
                    XcA.Add(i);
                    YcA.Add(j);
                    XpA.Add((int)projPixelA.x);
                    YpA.Add((int)projPixelA.y);

                }
                else // outside the shadow mask WE DONT CARE ABOUT THE PIXEL
                {
                    //Dont care is Light Gray
                    double[] color = { 0, 0, 0, 100 };
                    ProjPixMatA.put(j, i, color);
                }

            }

        }

        //Loop through the images from Cam B
        /**/
        for (int i = 0; i < cam_colsB; i++)
        {
            for (int j = 0; j < cam_rowsB; j++)
            {
                double[] colorNeutralOutsideMask = { 0, 0, 0, 100 };
                ProjPixMatB.put(j, i, colorNeutralOutsideMask);

                //if the pixel is not shadowed, reconstruct
                if (ShadowMaskB.get(j, i)[0] > 0)
                {

                    /**/ //projPixel style
                         //for a (x,y) pixel of the camera returns the corresponding projector pixel by calculating the decimal number
                    bool error = grayCode.getProjPixel(photosCamBGRAY, i, j, projPixelB);


                    if (error)
                    {
                        //Debug.Log("~~~Projpix error... ");
                        //Error (can't figure out what the pixel matches to, is Red, Error Pixel)
                          double[] colorE = { 255, 0, 0, 40 };
                           ProjPixMatB.put(j, i, colorE);
                        continue;
                    }
                    /**/
                    /*   END PRODUCT
                    Spit out image grayscale
                    CSV - 4 cols, Xc, Yc, Xp, Yp,
                    Rows - pixels of camera
                    */

                    double[] color = { 0, projPixelB.y / projectorPixHeight * 255, projPixelB.x / projectorPixWidth * 255, 255 };
                    ProjPixMatB.put(j, i, color);
                    XcB.Add(i);
                    YcB.Add(j);
                    XpB.Add((int)projPixelB.x);
                    YpB.Add((int)projPixelB.y);

                }
                else // outside the shadow mask WE DONT CARE ABOUT THE PIXEL
                {
                    //Dont care is dark gray
                    double[] color = { 0, 0, 0, 100 };
                    ProjPixMatB.put(j, i, color);
                }

            }

        }

        Debug.Log("~~~Showing Projector pixels mat... ");


        SaveMattoPNG(ProjPixMatA, "ProjPixMatA");
        SaveMattoPNG(ProjPixMatB, "ProjPixMatB");

        DisplayMatCamA(ProjPixMatA);
        DisplayMatCamB(ProjPixMatB);
        Debug.Log("~~~Saving Pixel data to CSV... ");
        SaveCSV();


        /**/



        //load cam1 intrinsics and cam 2 intrinsics


        //stereo rectify both images

        // photosCam1



        // Mat disparityMap;
        //List<Point> projPix = new List<Point>();


    }

    void saveSLData()
    {

        //TODO would be good to write the SLDATA automatically when capturing files

        /*
        FileStorage fs(outputFolder +"Stereo_intrinsics.yml", FileStorage::WRITE);
        if (fs.isOpened())
        {
            fs << "M1" << cameraMatrix << "D1" << distCoeffs <<
                "M2" << cameraMatrix1 << "D2" << distCoeffs1;
            fs.release();
        }
        else
            cout << "Error: can not save the intrinsic parameters\n";
        */

    }

    void SaveCSV()
    {
        string csvfilePath = getPath("ProjPointsCamA.CSV");

        //This is the writer, it writes to the filepath
        StreamWriter writer = new StreamWriter(csvfilePath);

        //This is writing the line of the Headser in the CSV
        writer.WriteLine("Xc,Yc,Xp,Yp");
        //This loops through everything sets the file to these.
        for (int i = 0; i < XcA.Count; ++i)
        {
            writer.WriteLine(XcA[i] +
                "," + YcA[i] +
                "," + XpA[i] +
                "," + YpA[i]);
        }
        writer.Flush();
        //This closes the file
        writer.Close();

        //Repeat for Cam b
        string csvfilePathB = getPath("ProjPointsCamB.CSV");

        //This is the writer, it writes to the filepath
        StreamWriter writerB = new StreamWriter(csvfilePathB);

        //This is writing the line of the Headser in the CSV
        writerB.WriteLine("Xc,Yc,Xp,Yp");
        //This loops through everything sets the file to these.
        for (int i = 0; i < XcB.Count; ++i)
        {
            writerB.WriteLine(XcB[i] +
                "," + YcB[i] +
                "," + XpB[i] +
                "," + YpB[i]);
        }
        writerB.Flush();
        //This closes the file
        writerB.Close();

    }
    private string getPath(string name)
    {
#if UNITY_EDITOR
        // return Application.dataPath + "/Graycode/" + "ProjPoints.csv";

        return "Graycode/" + name;
#elif UNITY_ANDROID
        return Application.persistentDataPath+"Saved_Inventory.csv";
#elif UNITY_IPHONE
        return Application.persistentDataPath+"/"+"Saved_Inventory.csv";
#else
        return Application.dataPath +"/"+"Saved_Inventory.csv";
#endif
    }
}
