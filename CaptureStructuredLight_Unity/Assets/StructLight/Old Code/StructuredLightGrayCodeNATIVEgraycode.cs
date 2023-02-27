
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

public class StructuredLightGrayCodeNATIVEgraycode : MonoBehaviour
{
    Params paramz;
    GrayCodePattern grayCode;

    List<Mat> pattern;
    List<Texture2D> patternTex;
    List<Mat> patternWB;
    List<Mat> photosCam1;
    List<Mat> photosCam2;
    List<Mat> photosCam1WB;//and all white and all black photos
    List<Mat> photosCam2WB;//and all white and all black photos
    public bool saveAllGrayCodes = true;
    Mat white;
    Mat black;

    /// <summary>
    /// The texture.
    /// </summary>
    Texture2D grayCodeTexture;

    int pattnum = 0;


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

    /// <summary>
    /// The rgba mat.
    /// </summary>
    Mat rgbaMat;

    /// <summary>
    /// The colors.
    /// </summary>
    Color32[] colors;

    /// <summary>
    /// Indicates whether this instance is waiting for initialization to complete.
    /// </summary>
    bool isInitWaiting = false;

    /// <summary>
    /// Indicates whether this instance has been initialized.
    /// </summary>
    bool hasInitDone = false;

    PhotoCapture photo1CaptureObject = null;
    Texture2D photo1TargetTexture = null;
    PhotoCapture photo2CaptureObject = null;
    Texture2D photo2TargetTexture = null;
    // Start is called before the first frame update
    void Start()
    {
        //Screen.SetResolution(projectorPixWidth, projectorPixHeight, true);

        pattern = new List<Mat>();
        patternTex = new List<Texture2D>();
        patternWB = new List<Mat>();
        photosCam1 = new List<Mat>();
        photosCam2 = new List<Mat>();
        photosCam1WB = new List<Mat>();
        photosCam2WB = new List<Mat>();

        black = new Mat();
        white = new Mat();
        paramz = new Params();

        Debug.Log("Projector Size W = " + projectorPixWidth + " H  = " + projectorPixHeight);

        grayCode = GrayCodePattern.create(projectorPixWidth, projectorPixHeight);

        grayCode.generate(pattern);

        Debug.Log("pattern size (OCV) = " + pattern.Count);

        GrayCodePatternUnity_Generate();
        Debug.Log("pattern size (UNITY) = " + patternTex.Count);

        grayCode.getImagesForShadowMasks(black, white);
        patternWB.Add(white);
        patternWB.Add(black);

        //pattern.Add(white);
        //pattern.Add(black);
        Debug.Log("patternWB size = " + patternWB.Count);

        //Debug Save all patterns as files
        if (saveAllGrayCodes == true)
        {
            for (int i = 0; i < pattern.Count; i++)
            {
                patternTex[i].Apply();
                byte[] bytes = patternTex[i].EncodeToPNG();
                string filePath = "Graycode/Graycode_" + 
                    i + ".png";
                File.WriteAllBytes(Application.absoluteURL + filePath, bytes);
                Debug.Log("Saving Graycode PNG: " + filePath);
            }
        }

        //  grayCodeTexture = new Texture2D( projectorPixWidth, projectorPixHeight, TextureFormat.ARGB32, true,true);


        grayCodeTexture = new Texture2D(projectorPixWidth, projectorPixHeight);//, TextureFormat.ARGB32, false);
       // GetComponent<RawImage>().material.mainTexture = grayCodeTexture;
        // gameObject.GetComponent<Renderer>().material.mainTexture = texture;

        //webcam stuff
        // webCamTexture = new WebCamTexture();
        // webCamTexture.Play();
        //colors = new Color32[webCamTexture.width * webCamTexture.height];
        //Photo capture stuff
        Debug.Log("Camera resolution" + PhotoCapture.SupportedResolutions.OrderByDescending((res) => res.width * res.height).First());
        Resolution cameraResolution = PhotoCapture.SupportedResolutions.OrderByDescending((res) => res.width * res.height).First();
        photo1TargetTexture = new Texture2D(cameraResolution.width, cameraResolution.height);

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
                // Take a picture
                photo1CaptureObject.TakePhotoAsync(OnCapturedPhotoToMemory);
            });
        });
        rgbaMat = new Mat(photo1TargetTexture.height, photo1TargetTexture.width, CvType.CV_8UC4);

    }


    void OnCapturedPhotoToMemory(PhotoCapture.PhotoCaptureResult result, PhotoCaptureFrame photoCaptureFrame)
    {
        // Copy the raw image data into our target texture
        photoCaptureFrame.UploadImageDataToTexture(photo1TargetTexture);

        // Create a gameobject that we can apply our texture to
        // GameObject quad = GameObject.CreatePrimitive(PrimitiveType.Quad);
        //Renderer quadRenderer = quad.GetComponent<Renderer>() as Renderer;
        //quadRenderer.material = new Material(Shader.Find("Unlit/Texture"));

        //quad.transform.parent = this.transform;
        //quad.transform.localPosition = new Vector3(0.0f, 0.0f, 3.0f);

        //quadRenderer.material.SetTexture("_MainTex", targetTexture);


        // Deactivate our camera
        // photoCaptureObject.StopPhotoModeAsync(OnStoppedPhotoMode);
    }

    void OnStoppedPhotoMode(PhotoCapture.PhotoCaptureResult result)
    {
        // Shutdown our photo capture resource
        photo1CaptureObject.Dispose();
        photo1CaptureObject = null;
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Backspace))
        {
            pattnum = 0;
            Debug.Log("Restarting Graycode calibration " + pattnum);


        }


        if (Input.GetKeyDown(KeyCode.Space))
        {
            if (pattnum < pattern.Count && pattnum > -1)
            {
                Debug.Log("Showing Graycode Pattern: " + pattnum);

                Mat thepattern = new Mat();
                pattern[pattnum].convertTo(thepattern, CvType.CV_8UC1);

                //Make Unity Display The graycode texture
                // Utils.matToTexture2D(pattern[pattnum], grayCodeTexture); // Opencv Way
                //Utils.textureToTexture2D(patternTex[pattnum], grayCodeTexture);
                  
                Texture2D theTexture = new Texture2D(projectorPixWidth, projectorPixHeight);
                //theTexture = patternTex[pattnum];
                //theTexture = patternTex.ElementAt<Texture2D>(pattnum);
                //grayCodeTexture = theTexture;

                // GetComponent<RawImage>().material.mainTexture.Equal(patternTex.ElementAt<Texture2D>(pattnum));
                //theTexture = patternTex[pattnum];
               // gameObject.GetComponent<CanvasRenderer>().SetTexture(grayCodeTexture);
                //Utils.matToTexture2D(thepattern, grayCodeTexture);


                // Utils.fastMatToTexture2D(pattern[pattnum], texture,true,0,false,false,false);
                gameObject.GetComponent<CanvasRenderer>().SetTexture(patternTex.ElementAt<Texture2D>(pattnum));
                // gameObject.GetComponent<RawImage>().material.mainTexture = patternTex.ElementAt<Texture2D>(pattnum);
                //patternTex.ElementAt<Texture2D>(pattnum).Apply();
                //gameObject.GetComponent<RawImage>().
                // GetComponent<RawImage>().material.mainTexture = grayCodeTexture;
                //gameObject.GetComponent<Renderer>().material.mainTexture = texture;

                //Debuggin saving raw pattern
                if (pattnum == 2*numOfColImgs-1 )
                {

                    byte[] bytes = grayCodeTexture.EncodeToPNG();
                    string filePath = "SavedScreen"+pattnum+".png";
                    File.WriteAllBytes(Application.absoluteURL + filePath, bytes);
                    Debug.Log("Saving PNG: " + filePath);

                }
                //Grab Photo after each Projection

                photo1CaptureObject.TakePhotoAsync(OnCapturedPhotoToMemory);

                Mat cam1Mat = new Mat(photo1TargetTexture.height, photo1TargetTexture.width, CvType.CV_8UC4);


                Utils.texture2DToMat(photo1TargetTexture, cam1Mat);

                photosCam1.Add(cam1Mat);
                Debug.Log("Take Photo from Cam 1: " + photosCam1.Count);

                // webCamTexture.Play();

                // webCamTexture.GetPixels32(colors);
                //  webCamTexture.Pause();
            }
            else if (pattnum < pattern.Count + 2)// Show the final all white and all black
            {
                Debug.Log("Showing WB Pattern: " + pattnum);
                Utils.matToTexture2D(patternWB[pattnum - pattern.Count], grayCodeTexture);
                gameObject.GetComponent<CanvasRenderer>().SetTexture(grayCodeTexture);



                photo1CaptureObject.TakePhotoAsync(OnCapturedPhotoToMemory);

                Mat cam1Mat = new Mat(photo1TargetTexture.height, photo1TargetTexture.width, CvType.CV_8UC4);
                Utils.texture2DToMat(photo1TargetTexture, cam1Mat);

                photosCam1WB.Add(cam1Mat);
                //Grab Photo after each Projection
                Debug.Log("Take photo from cam 1 WB: " + photosCam1WB.Count);
                if (pattnum < pattern.Count + 1)
                {
                    Debug.Log("Captured the White Image ");

                }
                else
                {
                    Debug.Log("Captured the Black Image ");
                    //That should be the final image, and we can process graycode i think!
                    ProcessGrayCodeImages();
                }
                //photosCam1.Add(cam1Mat);
            }
            pattnum++;
        }
        if (Input.GetKeyDown(KeyCode.W))
        {
            // gameObject.GetComponent<CanvasRenderer>().SetTexture(webCamTexture);
            //Texture2D texturego = new Texture2D();

            //Choose a rand photo to look at
            int photoRand = Random.Range(0, photosCam1.Count - 1);
            Debug.Log("photo to check " + photoRand);

            Utils.matToTexture2D(photosCam1[photoRand], photo1TargetTexture);
            // Take a picture
            gameObject.GetComponent<CanvasRenderer>().SetTexture(photo1TargetTexture);

        }
    }

    /*This is the c++ code for making gray code, trying to port it to native unity stuff to see if it is getting screwed up
     * 
     * 
     */
    void GrayCodePatternUnity_Generate()//List<Mat> pattern pattern,  )
    {

        numOfColImgs = Mathf.CeilToInt(Mathf.Log((float)projectorPixWidth) / Mathf.Log((float)2.0));
        numOfRowImgs = Mathf.CeilToInt(Mathf.Log((float)projectorPixHeight) / Mathf.Log((float)2.0));
        // numOfRowImgs = ceil(log(double( params.height)) / log(2.0));
        int numOfPatternImages = 2 * numOfColImgs + 2 * numOfRowImgs;

        for (int i = 0; i < pattern.Count; i++) //load it up with blank patterns
        {
       //     Texture2D patTexture = new Texture2D(projectorPixWidth, projectorPixHeight);
       
            // patternTex.Add(patTexture);

            patternTex.Add(new Texture2D(projectorPixWidth, projectorPixHeight));
        // pattern_[i] = Mat( params.height, params.width, CV_8U);
        }

        int flagg = 0;
        for (int j = 0; j < projectorPixHeight; j++)   //Go through the cols Loop
        {

            int rem = 0, num = j, prevRem = j % 2;


            for (int k = 0; k < numOfColImgs; k++)  // images loop
            {
                num = num / 2;
                rem = num % 2;

                if ((rem == 0 && prevRem == 1) || (rem == 1 && prevRem == 0))
                {
                    flagg = 1;
                }
                else
                {
                    flagg = 0;
                }

                for (int i = 0; i < projectorPixWidth; i++)  // rows loop
                {

                    Color pixel_color = Color.red; //Initialize
                    Color pixel_color_INV = Color.green; //Initialize
                    if (flagg == 1)
                    {
                        pixel_color = Color.green;
                        pixel_color_INV = Color.blue;

                    }
                    else
                    {
                        pixel_color = Color.black;
                        pixel_color_INV = Color.white;

                    }

                    patternTex[2 * numOfColImgs - 2 * k - 2].SetPixel(i, j, pixel_color);
                    patternTex[2 * numOfColImgs - 2 * k - 1].SetPixel(i, j, pixel_color_INV); //inverse image

                }

                prevRem = rem;
            }
        }

        for (int i = 0; i < projectorPixWidth; i++)  // rows loop
        {
            int rem = 0, num = i, prevRem = i % 2;

            for (int k = 0; k < numOfRowImgs; k++) // Images loop
            {
                num = num / 2;
                rem = num % 2;

                if ((rem == 0 && prevRem == 1) || (rem == 1 && prevRem == 0))
                {
                    flagg = 1;
                }
                else
                {
                    flagg = 0;
                }

                for (int j = 0; j < projectorPixHeight; j++)  //Cols loop
                {

                    Color pixel_color = Color.red; //Initialize
                    Color pixel_color_INV = Color.green; //Initialize
                    if (flagg == 1)
                    {
                        pixel_color = Color.white;
                        pixel_color_INV = Color.black;

                    }
                    else
                    {
                        pixel_color = Color.black;
                        pixel_color_INV = Color.white;

                    }
                    patternTex[2 * numOfRowImgs - 2 * k + 2 * numOfColImgs - 2].SetPixel(i, j, pixel_color);
                   // patternTex[2 * numOfRowImgs - 2 * k + 2 * numOfColImgs - 2].Apply();
                    patternTex[2 * numOfRowImgs - 2 * k + 2 * numOfColImgs - 1].SetPixel(i, j, pixel_color_INV);
                    //patternTex[2 * numOfRowImgs - 2 * k + 2 * numOfColImgs - 1].Apply();

                    //uchar pixel_color = (uchar)flag * 255;
                    //  pattern_[2 * numOfRowImgs - 2 * k + 2 * numOfColImgs - 2].at<uchar>(i, j) = pixel_color;

                }

                prevRem = rem;
            }
        }
   
      

    }


    /*original stuff
     * 
    std::vector<Mat> & pattern_ = *(std::vector<Mat>*)pattern.getObj();
    pattern_.resize(numOfPatternImages);

    for (size_t i = 0; i < numOfPatternImages; i++)
    {
        pattern_[i] = Mat( params.height, params.width, CV_8U);
    }

    uchar flag = 0;

    for (int j = 0; j < params.width; j++ )  // rows loop
{
        int rem = 0, num = j, prevRem = j % 2;

        for (size_t k = 0; k < numOfColImgs; k++)  // images loop
        {
            num = num / 2;
            rem = num % 2;

            if ((rem == 0 && prevRem == 1) || (rem == 1 && prevRem == 0))
            {
                flag = 1;
            }
            else
            {
                flag = 0;
            }

            for (int i = 0; i < params.height; i++ )  // rows loop
  {

            uchar pixel_color = (uchar)flag * 255;

            pattern_[2 * numOfColImgs - 2 * k - 2].at<uchar>(i, j) = pixel_color;
            if (pixel_color > 0)
                pixel_color = (uchar)0;
            else
                pixel_color = (uchar)255;
            pattern_[2 * numOfColImgs - 2 * k - 1].at<uchar>(i, j) = pixel_color;  // inverse
        }

        prevRem = rem;
    }
}

for(int i = 0; i< params.height; i++ )  // rows loop
{
int rem = 0, num = i, prevRem = i % 2;

for(size_t k = 0; k<numOfRowImgs; k++ )
{
  num = num / 2;
  rem = num % 2;

  if((rem == 0 && prevRem == 1) || (rem == 1 && prevRem == 0) )
  {
    flag = 1;
  }
  else
  {
    flag = 0;
  }

  for(int j = 0; j< params.width; j++ )
  {

    uchar pixel_color = (uchar)flag * 255;
pattern_[2 * numOfRowImgs - 2 * k + 2 * numOfColImgs - 2].at<uchar>(i, j) = pixel_color;

    if(pixel_color > 0 )
      pixel_color = (uchar ) 0;
    else
      pixel_color = (uchar ) 255;

    pattern_[2 * numOfRowImgs - 2 * k + 2 * numOfColImgs - 1].at<uchar>(i, j) = pixel_color;
  }

  prevRem = rem;
}
}


*/





    void ProcessGrayCodeImages()
    {

        //load cam1 intrinsics and cam 2 intrinsics


        //stereo rectify both images

        // photosCam1



        Mat disparityMap;
        List<Point> projPix = new List<Point>();

        for (int x = 0; x < 1920; x++)
        {
            for (int y = 0; y < 1080; y++)
            {


                Point thepoint = new Point(-1, -1);
                //grayCode.decode()
                Debug.Log(thepoint);
                grayCode.getProjPixel(photosCam1, x, y, thepoint);
                projPix.Add(thepoint);
            }

        }
        Debug.Log("GetprojPrixel Complete " + projPix.Count);
        MatOfPoint matofpoint = new MatOfPoint();
        //matofpoint.
        // bool decoded= Structured_light.DECODE_3D_UNDERWORLD()

        /*
         * 
         * 
         * 
         *   // Loading calibration parameters
  Mat cam1intrinsics, cam1distCoeffs, cam2intrinsics, cam2distCoeffs, R, T;
  fs["cam1_intrinsics"] >> cam1intrinsics;
  fs["cam2_intrinsics"] >> cam2intrinsics;
  fs["cam1_distorsion"] >> cam1distCoeffs;
  fs["cam2_distorsion"] >> cam2distCoeffs;
  fs["R"] >> R;
  fs["T"] >> T;
  cout << "cam1intrinsics" << endl << cam1intrinsics << endl;
  cout << "cam1distCoeffs" << endl << cam1distCoeffs << endl;
  cout << "cam2intrinsics" << endl << cam2intrinsics << endl;
  cout << "cam2distCoeffs" << endl << cam2distCoeffs << endl;
  cout << "T" << endl << T << endl << "R" << endl << R << endl;
  if( (!R.data) || (!T.data) || (!cam1intrinsics.data) || (!cam2intrinsics.data) || (!cam1distCoeffs.data) || (!cam2distCoeffs.data) )
  {
    cout << "Failed to load cameras calibration parameters" << endl;
    help();
    return -1;
  }
  size_t numberOfPatternImages = graycode->getNumberOfPatternImages();
  vector<vector<Mat> > captured_pattern;
  captured_pattern.resize( 2 );
  captured_pattern[0].resize( numberOfPatternImages );
  captured_pattern[1].resize( numberOfPatternImages );
  Mat color = imread( imagelist[numberOfPatternImages], IMREAD_COLOR );
  Size imagesSize = color.size();
  // Stereo rectify
  cout << "Rectifying images..." << endl;
  Mat R1, R2, P1, P2, Q;
  Rect validRoi[2];
  stereoRectify( cam1intrinsics, cam1distCoeffs, cam2intrinsics, cam2distCoeffs, imagesSize, R, T, R1, R2, P1, P2, Q, 0,
                -1, imagesSize, &validRoi[0], &validRoi[1] );
  Mat map1x, map1y, map2x, map2y;
  initUndistortRectifyMap( cam1intrinsics, cam1distCoeffs, R1, P1, imagesSize, CV_32FC1, map1x, map1y );
  initUndistortRectifyMap( cam2intrinsics, cam2distCoeffs, R2, P2, imagesSize, CV_32FC1, map2x, map2y );
  // Loading pattern images
  for( size_t i = 0; i < numberOfPatternImages; i++ )
  {
    captured_pattern[0][i] = imread( imagelist[i], IMREAD_GRAYSCALE );
    captured_pattern[1][i] = imread( imagelist[i + numberOfPatternImages + 2], IMREAD_GRAYSCALE );
    if( (!captured_pattern[0][i].data) || (!captured_pattern[1][i].data) )
    {
      cout << "Empty images" << endl;
      help();
      return -1;
    }
    remap( captured_pattern[1][i], captured_pattern[1][i], map1x, map1y, INTER_NEAREST, BORDER_CONSTANT, Scalar() );
    remap( captured_pattern[0][i], captured_pattern[0][i], map2x, map2y, INTER_NEAREST, BORDER_CONSTANT, Scalar() );
  }
  cout << "done" << endl;
  vector<Mat> blackImages;
  vector<Mat> whiteImages;
  blackImages.resize( 2 );
  whiteImages.resize( 2 );
  // Loading images (all white + all black) needed for shadows computation
  cvtColor( color, whiteImages[0], COLOR_RGB2GRAY );
  whiteImages[1] = imread( imagelist[2 * numberOfPatternImages + 2], IMREAD_GRAYSCALE );
  blackImages[0] = imread( imagelist[numberOfPatternImages + 1], IMREAD_GRAYSCALE );
  blackImages[1] = imread( imagelist[2 * numberOfPatternImages + 2 + 1], IMREAD_GRAYSCALE );
  remap( color, color, map2x, map2y, INTER_NEAREST, BORDER_CONSTANT, Scalar() );
  remap( whiteImages[0], whiteImages[0], map2x, map2y, INTER_NEAREST, BORDER_CONSTANT, Scalar() );
  remap( whiteImages[1], whiteImages[1], map1x, map1y, INTER_NEAREST, BORDER_CONSTANT, Scalar() );
  remap( blackImages[0], blackImages[0], map2x, map2y, INTER_NEAREST, BORDER_CONSTANT, Scalar() );
  remap( blackImages[1], blackImages[1], map1x, map1y, INTER_NEAREST, BORDER_CONSTANT, Scalar() );
  cout << endl << "Decoding pattern ..." << endl;
  Mat disparityMap;
  bool decoded = graycode->decode( captured_pattern, disparityMap, blackImages, whiteImages,
                                  structured_light::DECODE_3D_UNDERWORLD );
  if( decoded )
  {
    cout << endl << "pattern decoded" << endl;
    // To better visualize the result, apply a colormap to the computed disparity
    double min;
    double max;
    minMaxIdx(disparityMap, &min, &max);
    Mat cm_disp, scaledDisparityMap;
    cout << "disp min " << min << endl << "disp max " << max << endl;
    convertScaleAbs( disparityMap, scaledDisparityMap, 255 / ( max - min ) );
    applyColorMap( scaledDisparityMap, cm_disp, COLORMAP_JET );
    // Show the result
    resize( cm_disp, cm_disp, Size( 640, 480 ), 0, 0, INTER_LINEAR_EXACT );
    imshow( "cm disparity m", cm_disp );
    // Compute the point cloud
    Mat pointcloud;
    disparityMap.convertTo( disparityMap, CV_32FC1 );
    reprojectImageTo3D( disparityMap, pointcloud, Q, true, -1 );
    // Compute a mask to remove background
    Mat dst, thresholded_disp;
    threshold( scaledDisparityMap, thresholded_disp, 0, 255, THRESH_OTSU + THRESH_BINARY );
    resize( thresholded_disp, dst, Size( 640, 480 ), 0, 0, INTER_LINEAR_EXACT );
    imshow( "threshold disp otsu", dst );
#ifdef HAVE_OPENCV_VIZ
    // Apply the mask to the point cloud
    Mat pointcloud_tresh, color_tresh;
    pointcloud.copyTo( pointcloud_tresh, thresholded_disp );
    color.copyTo( color_tresh, thresholded_disp );
    // Show the point cloud on viz
    viz::Viz3d myWindow( "Point cloud with color" );
    myWindow.setBackgroundMeshLab();
    myWindow.showWidget( "coosys", viz::WCoordinateSystem() );
    myWindow.showWidget( "pointcloud", viz::WCloud( pointcloud_tresh, color_tresh ) );
    myWindow.showWidget( "text2d", viz::WText( "Point cloud", Point(20, 20), 20, viz::Color::green() ) );
    myWindow.spin();
#endif // HAVE_OPENCV_VIZ
  }
  waitKey();
  return 0;
}
         */
    }
}