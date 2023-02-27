/*
Dual Camera Capturer

Inputs:
- 2 web cams connected to the compupter


This program will take stereo pairs


The program runs with visual feedback to someone with the chessboard to make sure the program is capturing good images
the program will also run on an automatic timer to assist with calibration if you don't have any friends :)
*/


#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <iostream>
#include <ctime>

using namespace std;
using namespace cv;

#define KEY_UP 72
#define KEY_DOWN 80
#define KEY_LEFT 75
#define KEY_RIGHT 77

// THIS FILE SHOULD NOT CARE ABOUT CHESSBOARDS
//big board -w=6 -h=5 -sl=0.05 -ml=0.037
//Real big board w = 10    h = 13   sl = .06
namespace {
	const char* about =
		"Calibration using a CHESS board\n"
		"  To capture a frame for calibration, press 'c',\n"
		"  If input comes from video, press any key for next frame\n"
		"  To finish capturing, press 'ESC' key and calibration starts.\n";
	const char* keys =
		"{w        |   6    | Number of squares in X direction }"
		"{h        |   5   | Number of squares in Y direction }"
		"{sl       |    .05   | Square side length (in meters) }"
		"{v        |       | Input from video file, if ommited, input comes from camera }"

		"{ciA       | 0     | Camera id if input doesnt come from video (-v) }"
		"{ciB       | 1    | Camera id if input doesnt come from video (-v) }"

		"{dp       |       | File of marker detector parameters }"
		"{rs       | true | Apply refind strategy }"
		"{zt0       | false | Assume zero tangential distortion }"
		"{zt1       | false | Assume zero tangential distortion }"

		"{a0        |       | Fix aspect ratio (fx/fy) to this value }"
		"{a1        |       | Fix aspect ratio (fx/fy) to this value }"

		"{pc0       | false | Fix the principal point at the center }"
		"{pc1       | false | Fix the principal point at the center }"

		"{sc       | false | Show detected chessboard corners after calibration }"

		"{loadNumImgs       | 0 | number of images to use in the Calibration from Files - If 0 use live web cameras }"
		"{camAfilename       | camA_im | prefix for Cam A images }"
		"{camBfilename       | camB_im | prefix for Cam B images }"
		"{fileExtension       | .png | suffix for both types of images }"
		;
}


/**
* Function to save all the parameters to a file
 */
static bool saveCameraParams(const string& filename, Size imageSize, float aspectRatio, int flags,
	const Mat& cameraMatrix, const Mat& distCoeffs, double totalAvgErr) {
	FileStorage fs(filename, FileStorage::WRITE);
	if (!fs.isOpened())
		return false;

	time_t tt;
	time(&tt);
	struct tm* t2 = localtime(&tt);
	char buf[1024];
	strftime(buf, sizeof(buf) - 1, "%c", t2);

	fs << "calibration_time" << buf;

	fs << "image_width" << imageSize.width;
	fs << "image_height" << imageSize.height;

	if (flags & CALIB_FIX_ASPECT_RATIO) fs << "aspectRatio" << aspectRatio;

	if (flags != 0) {
		sprintf(buf, "flags: %s%s%s%s",
			flags & CALIB_USE_INTRINSIC_GUESS ? "+use_intrinsic_guess" : "",
			flags & CALIB_FIX_ASPECT_RATIO ? "+fix_aspectRatio" : "",
			flags & CALIB_FIX_PRINCIPAL_POINT ? "+fix_principal_point" : "",
			flags & CALIB_ZERO_TANGENT_DIST ? "+zero_tangent_dist" : "");
	}

	fs << "flags" << flags;

	fs << "camera_matrix" << cameraMatrix;
	fs << "distortion_coefficients" << distCoeffs;

	fs << "avg_reprojection_error" << totalAvgErr;

	return true;
}

/**
* Function to save all the STEREO parameters to a file
 */
static bool saveCameraParamsStereo(const string& filename, Size imageSize, Size imageSize1, float aspectRatio, float aspectRatio1, int flags, int flags1,
	const Mat& cameraMatrix, const Mat& cameraMatrix1, const Mat& distCoeffs, const Mat& distCoeffs1, double totalAvgErr, double totalAvgErr1,
	Mat R, Mat T, Mat R1, Mat R2, Mat P1, Mat P2, Mat Q, double StereoRMS, double exposureA, double exposureB, double focusA, double focusB, int numSquaresX, int numSquaresY, double SquareSideLength) {
	FileStorage fs(filename, FileStorage::WRITE);
	if (!fs.isOpened())
		return false;

	time_t tt;
	time(&tt);
	struct tm* t2 = localtime(&tt);
	char buf[1024];
	strftime(buf, sizeof(buf) - 1, "%c", t2);

	fs << "camA_intrinsics" << cameraMatrix;

	fs << "camA_distorsion" << distCoeffs;

	fs << "camA_size" << imageSize;
	fs << "camA_error" << totalAvgErr;


	fs << "camB_intrinsics" << cameraMatrix1;

	fs << "camB_distorsion" << distCoeffs1;

	fs << "camB_size" << imageSize;
	fs << "camB_error" << totalAvgErr1;


	fs << "R" << R;
	fs << "T" << T;
	fs << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
	fs << "stereo_error" << StereoRMS;

	//Camera Info
	fs << "ExposureA" << exposureA << "ExposureB" << exposureB << "FocusA" << focusA << "FocusB" << focusB;
	//Chessboard info
	fs << "ChessNumSquaresX" << numSquaresX << "ChessNumSquaresY" << numSquaresY << "ChessSquareSideLength" << SquareSideLength;

	fs << "calibration_time" << buf;

	return true;
}


/**
 */
int main(int argc, char* argv[]) {

	//Toggle to apply 1/4 scaling to the images when finding chessboards (and then scaling back up later)
	bool downsampleChess = false;

	//Check on our camera performance
	double realfps = 1;


	/* LOAD DATA
	...............Parse all the user's information................
	*/
	CommandLineParser parser(argc, argv, keys);
	parser.about(about);

	if (argc < 7) {
		parser.printMessage();
		return 0;
	}

	int squaresX = parser.get<int>("w");
	int squaresY = parser.get<int>("h");
	float squareLength = parser.get<float>("sl");
	//string outputFolder = parser.get<string>("output");
	string outputFolder = "C:/Users/andre/Desktop/Glowcake Hoss/Scans";
	bool showChessboardCorners = parser.get<bool>("sc");

	int calibrationFlagsA = 0;
	int calibrationFlagsB = 0;

	float aspectRatio = 1;
	float aspectRatio1 = 1;

	int focuscamA = 0;
	int focuscamB = 0;

	int expcamA = -5;
	int expcamB = -5;


	int camIdA = parser.get<int>("ciA");
	int camIdB = parser.get<int>("ciB");
	String video;

	String camAfilename = parser.get<String>("camAfilename", true);
	String camBfilename = parser.get<String>("camBfilename", true);
	String fileExtension = parser.get<String>("fileExtension", true);


	int loadNumImgs = parser.get<float>("loadNumImgs");

	if (!parser.check()) {
		parser.printErrors();
		return 0;
	}

	cout << "Initialize Params and or Load Images" << endl;

	//cout << "Intialize Boards" << endl;
	//Set up Chessboard Detection
	Size board_size = Size(squaresX, squaresY);
	//cout << "board is " << squaresX << "  by  " << squaresY << endl;

	// collect data from each frame
	vector< Mat > allImgsA;
	vector< Mat > allImgsB;
	Size imgSizeA;
	Size imgSizeB;

	vector< vector< Point2f > >  imagePointsA, imagePointsB;
	vector< vector< vector< Point2f > > > allCorners1;

	vector< vector< Point3f > > objectPointsA, objectPointsB; //This is a bit silly, there is only one object, this objectpointsB is just the same as A
	vector< vector< int > > allIds1;

	bool useSBforfindingCorners = false;
	bool foundAq = false;
	bool foundBq = false;
	vector< Point2f > cornersAq, cornersBq;

	VideoCapture inputVideoA;
	VideoCapture inputVideoB;

	//timer stuff
	time_t prev_time = time(0);
	int countdown_time = 3;


	//LOAD IMAGES FROM FILE
	//Only do this if the number isn't 0. If the number is 0 we will capture new images from the webcams
	if (loadNumImgs > 0) {
		cout << "LOAD IMAGES FROM Stereo FILES " << loadNumImgs << endl;

		for (int i = 0; i < loadNumImgs; i++) { //Load images and DON"T USE LIVE CAMERA
			//char A_img[100], B_img[100];
			//sprintf(A_img, "%s%s%d.%s", outputFolder, camAfilename, i, fileExtension);
			//sprintf(B_img, "%s%s%d.%s", outputFolder, camBfilename, i, fileExtension);
			Mat imgA, imgB;
			string A_img = outputFolder + "/" + camAfilename + to_string(i) + fileExtension;
			string B_img = outputFolder + "/" + camBfilename + to_string(i) + fileExtension;

			imgA = imread(A_img, IMREAD_COLOR);
			imgB = imread(B_img, IMREAD_COLOR);

			cout << "Frame " << i << " loaded camA  " << A_img << "  |  ";

			allImgsA.push_back(imgA);
			imgSizeA = imgA.size();
			cout << "Frame " << i << " loaded camB   " << B_img << endl;

			allImgsB.push_back(imgB);
			imgSizeB = imgB.size();


			/// Double check our loaded images
			//Shrink the Image for Display purposes
			Size showsize;
			//showsize = Size(960, 540);
			showsize = Size(640, 480);
			Mat imageCopyLowResA, imageCopyLowResB;

			imgA.copyTo(imageCopyLowResA);
			imgB.copyTo(imageCopyLowResB);
			resize(imageCopyLowResA, imageCopyLowResA, showsize, 0, 0);
			resize(imageCopyLowResB, imageCopyLowResB, showsize, 0, 0);


			//Detect Chessboards // THis is just double checking the images have a valid chessboard on them, that's why we use FAST CHECK
			//foundAq = cv::findChessboardCorners(imageCopyLowResA, board_size, cornersAq, CALIB_CB_FAST_CHECK);
			//foundBq = cv::findChessboardCorners(imageCopyLowResB, board_size, cornersBq, CALIB_CB_FAST_CHECK);

			if (!foundAq || !foundBq) //skip if we dont get a chessboard, extra check
			{
				cout << "small size error on " << i << "   no chessboard found.   found a and b   " << foundAq << foundBq << endl;
			}
			else
			{
				cout << "Found Board small size  " << i << endl;
				drawChessboardCorners(imageCopyLowResA, board_size, cornersAq, foundAq);
				imshow("test loaded imgs", imageCopyLowResA);
				waitKey(1);
			}

		}
		//imshow("test", allImgsA[0]);
		cout << "Image Sizes " << imgSizeA << "   images b   " << imgSizeB << endl;
	}

	// WEBCAM LIVE IMAGES
	//Do Live Image Capture
	else {
		int waitTime;

		if (!video.empty()) { // legacy function if we wanted to get frames from a video file
			cout << "Capture from Video Frames " << loadNumImgs << endl;
			inputVideoA.open(video);
			waitTime = 0;
		}
		else {
			cout << "Live Capture Images " << loadNumImgs << endl;

			inputVideoA.open(camIdA, CAP_DSHOW);
			inputVideoB.open(camIdB, CAP_DSHOW);
			//inputVideoA.open(camIdA);
			//inputVideoB.open(camIdB); //  runs a bit faster without DSHOW
			/*inputVideoA.open(camIdA, CAP_INTEL_MFX);
			inputVideoB.open(camIdB, CAP_INTEL_MFX);/**/

			waitTime = 2;
		}



		//inputVideoA.set(CAP_PROP_FOURCC, VideoWriter::fourcc('H' , '2', '6', '4'));			//Camera Settings Dialog

		//This pops up the nice dialog to keep camera settings persistent. You need directshow DSHOW enabled as the capturer, and you need a number here that doesn't do anything but you have to have it there
		inputVideoA.set(CAP_PROP_SETTINGS, 0);
		inputVideoB.set(CAP_PROP_SETTINGS, 0);
		//inputVideoA.set(CAP_PROP_MONOCHROME, 1);





		//Manually Set Camera Parameters

			//Programmatically set Exposure
		inputVideoA.set(CAP_PROP_EXPOSURE, expcamA);
		inputVideoB.set(CAP_PROP_EXPOSURE, expcamB);
		//inputVideoA.set(CAP_PROP_AUTO_EXPOSURE, .25);

		//inputVideoA.set(CAP_PROP_FPS, 5);
		//inputVideoB.set(CAP_PROP_FPS, 5);

		/**/ // Brio Cameras 3840 2160 // Things says brio can go up to 4096 x 2160 pixels for better locking on to 4k// Old 4k cams 3264 2448
		inputVideoA.set(CAP_PROP_FRAME_WIDTH, 4096);
		inputVideoA.set(CAP_PROP_FRAME_HEIGHT, 2160);
		inputVideoB.set(CAP_PROP_FRAME_WIDTH, 4096);
		inputVideoB.set(CAP_PROP_FRAME_HEIGHT, 2160);
		/**/

		inputVideoA.set(CAP_PROP_FOCUS, focuscamA);
		inputVideoB.set(CAP_PROP_FOCUS, focuscamB);

		//TODO Maybe give it a little little delay to make sure the Set takes

		cout << "Cameras Started" << endl;
		cout << "Cameras A Properties " << " ID num " << camIdA << " exposure " << inputVideoA.get(CAP_PROP_EXPOSURE) << " Focus " << inputVideoA.get(CAP_PROP_FOCUS) << "  Backend API " << inputVideoA.get(CAP_PROP_BACKEND) << "  Width and Height " << inputVideoA.get(CAP_PROP_FRAME_WIDTH) << " " << inputVideoA.get(CAP_PROP_FRAME_HEIGHT) << endl;
		cout << "Cameras B Properties " << " ID num " << camIdB << " exposure " << inputVideoB.get(CAP_PROP_EXPOSURE) << " Focus " << inputVideoB.get(CAP_PROP_FOCUS) << "  Backend API " << inputVideoB.get(CAP_PROP_BACKEND) << "  Width and Height " << inputVideoB.get(CAP_PROP_FRAME_WIDTH) << " " << inputVideoB.get(CAP_PROP_FRAME_HEIGHT) << endl;

		cout << "Create Windows" << endl;
		namedWindow("CamA_StereoCalib_Output", WINDOW_KEEPRATIO);
		moveWindow("CamA_StereoCalib_Output", 0, 10);
		resizeWindow("CamA_StereoCalib_Output", 1920, 540);
		cout << "Press Esc key to save all images" << endl;

		//How many frames we have captured
		int framenum = 0;

		//This is the main Frame-grabbing loop
		while (1) //inputVideoA.grab() && inputVideoB.grab()) // alternative method to grab frames at the same time! for multicam
		{

			//Start the FPS timer
			int64 tickstart = cv::getTickCount();
			time_t current_time = time(0) - prev_time;

			Mat imageA, imageCopyLowResA, grayA;
			if (inputVideoA.isOpened())
			{
				Mat viewA;
				inputVideoA >> viewA;
				viewA.copyTo(imageA);
			}
			//inputVideoA.retrieve(imageA);

			Mat imageB, imageCopyLowResB, grayB;
			if (inputVideoB.isOpened())
			{
				Mat viewB;
				inputVideoB >> viewB;
				viewB.copyTo(imageB);
			}
			//inputVideoB.retrieve(imageB);

			//inputVideoA >> imageA;
			//inputVideoB >> imageB;

			vector< int > ids;
			vector< Point2f > cornersA, cornersB, rejected;

			vector< int > ids1;
			vector< vector< Point2f > > corners1, rejected1;


			//First search for corners at LOW RES while live streaming, this is just to make sure we are only capturing good frames and giving feedback

			//Shrink the Image for Display purposes
			Size showsize;
			//showsize = Size(960, 540);
			showsize = Size(640, 480);

			imageA.copyTo(imageCopyLowResA);
			imageB.copyTo(imageCopyLowResB);
			resize(imageCopyLowResA, imageCopyLowResA, showsize, 0, 0);
			resize(imageCopyLowResB, imageCopyLowResB, showsize, 0, 0);



			//Detect Chessboards

			bool foundA = false;
			//foundA = cv::findChessboardCorners(imageCopyLowResA, board_size, cornersA, CALIB_CB_FAST_CHECK);

			bool foundB = false;
			//	foundB = cv::findChessboardCorners(imageCopyLowResB, board_size, cornersB, CALIB_CB_FAST_CHECK);

			if (foundA)
			{
				//cornerSubPix(grayA, corners, cv::Size(5, 5), cv::Size(-1, -1), 				TermCriteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.1));
				drawChessboardCorners(imageCopyLowResA, board_size, cornersA, foundA);
			}
			if (foundB)
			{
				//	cornerSubPix(grayB, corners, cv::Size(5, 5), cv::Size(-1, -1), 				TermCriteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.1));
				drawChessboardCorners(imageCopyLowResB, board_size, cornersB, foundB);
			}

			Scalar texColA = Scalar(255, 0, 0);
			Scalar texColB = Scalar(255, 0, 0);

			if (!foundA) {
				texColA = Scalar(0, 0, 255);
				putText(imageCopyLowResA, "NOT ALL POINTS VISIBLE ",
					Point(10, 100), FONT_HERSHEY_SIMPLEX, 1.4, texColA, 4);

				//show red detected dots
				drawChessboardCorners(imageCopyLowResA, board_size, cornersA, foundA);
			}
			putText(imageCopyLowResA, "Cam A: Press 'c' to add current frame. 'ESC' to finish and calibrate",
				Point(10, 20), FONT_HERSHEY_SIMPLEX, .5, texColA, 2);


			if (!foundB) {
				texColB = Scalar(0, 0, 255);

				putText(imageCopyLowResB, "NOT ALL POINTS VISIBLE ",
					Point(10, 100), FONT_HERSHEY_SIMPLEX, 1.4, texColB, 2);

				//show red detected dots
				drawChessboardCorners(imageCopyLowResB, board_size, cornersB, foundB);
			}
			putText(imageCopyLowResB, "Cam B: 'c'=add current frame. 'ESC'= calibrate",
				Point(10, 20), FONT_HERSHEY_SIMPLEX, .5, texColB, 2);

			//Show the FPS
			putText(imageCopyLowResA, "FPS: " + to_string(realfps) + "   Cap Frame No: " + to_string(framenum),
				Point(10, 400), FONT_HERSHEY_SIMPLEX, 1, texColA, 2);
			putText(imageCopyLowResB, "FPS: " + to_string(realfps) + "   Cap Frame No: " + to_string(framenum),
				Point(10, 400), FONT_HERSHEY_SIMPLEX, 1, texColA, 2);

			//Put windows next to each other
			hconcat(imageCopyLowResA, imageCopyLowResB, imageCopyLowResA);
			imshow("CamA_StereoCalib_Output", imageCopyLowResA);

			//HANDLE USER INPUT 
			char key = (char)waitKey(waitTime);

			//Leave this loop if we hit escape
			if (key == 27) {


				//Save all the Captured Images, keep them in the vault
				cout << "Saving All images" << endl;

				bool save1 = false;
				bool save2 = false;

				for (int i = 0; i < allImgsA.size(); i++) {
					ostringstream name;
					//name << i + 1;
					name << i;
					save1 = imwrite(outputFolder + "/" + "camA_im" + name.str() + ".png", allImgsA[i]);
					save2 = imwrite(outputFolder + "/" + "camB_im" + name.str() + ".png", allImgsB[i]);
					if ((save1) && (save2))
					{
						cout << "pattern camA and camB images number " << i << " saved" << endl << endl;

					}
					else
					{
						cout << "pattern camA and camB images number " << i << " NOT saved" << endl << endl << "Retry, check the path" << endl << endl;
					}
				}
				//Kill the Cameras
				inputVideoA.release();
				inputVideoB.release();
				break;
			}

			//Change Camera around if we need
			if (key == '0') {
				inputVideoA.open(0);
				cout << "Changed CamA to Input 0 " << endl;
			}
			if (key == '1') {
				inputVideoA.open(1);
				cout << "Changed CamA to Input 1 " << endl;
			}
			if (key == '2') {
				inputVideoA.open(2);
				cout << "Changed CamA to Input 2 " << endl;
			}
			if (key == '3') {
				inputVideoA.open(3);
				cout << "Changed CamA to Input 3 " << endl;
			}

			if (key == KEY_LEFT) {
				inputVideoB.open(0);
				cout << "Changed CamB to Input 0 " << endl;
			}
			if (key == KEY_DOWN) {
				inputVideoB.open(1);
				cout << "Changed CamB to Input 1 " << endl;
			}
			if (key == KEY_RIGHT) {
				inputVideoB.open(2);
				cout << "Changed CamB to Input 2 " << endl;
			}

			//trigger via time every X seconds
			if (current_time > countdown_time) {
				prev_time = time(0);
				key = 'c'; //Enable or disable countdown timer
				cout << '\a';
			}

			//Manually capture when user presses C button
		/*	if (key == 'c' && (!foundA || !foundB)) {
				cout << "Frame Not Captured, Please make sure all IDs are visible!" << endl;
				putText(imageCopyLowResB, "BAD CAPTURE",
					Point(10, 20), FONT_HERSHEY_SIMPLEX, .5, texColB, 2);
			}*/

			//Process the frame that has a chessboard in it
			//if (key == 'c' && foundA && foundB) {
			if (key == 'c') {

				putText(imageCopyLowResB, "CAPTURED FRAME",
					Point(10, 20), FONT_HERSHEY_SIMPLEX, .5, texColB, 2);

				//Process the Captured Frame Chess corners
				//Cam A
				cout << "Frame " << framenum << " captured camA" << endl;
				allImgsA.push_back(imageA);
				imgSizeA = imageA.size();

				//Cam B
				cout << "Frame captured camB" << endl;
				allImgsB.push_back(imageB);
				imgSizeB = imageB.size();

				framenum++;
			}
			//Calculate Framerate
			realfps = cv::getTickFrequency() / (cv::getTickCount() - tickstart);
		}
	}


	/*
	******
	PROCESS STAGE
	Take all the images and do a thorough search for chessboards and then calibrate!
	******
	*/

	vector< Point2f > cornersA, cornersB;

	//Calibrate those chessboards individually!
	cout << "Calibrating Cam A and Cam B at Full Resolution | total images= " << allImgsA.size() << endl;
	for (int i = 0; i < allImgsA.size(); i++) {

		if (downsampleChess) { // Downscale everything by 1/4 and then upscale
			cout << "DOWNSAMPLING and upsampling later" << endl;
			resize(allImgsA[i], allImgsA[i], Size(), 0.25, 0.25);
			resize(allImgsB[i], allImgsB[i], Size(), 0.25, 0.25);
		}

		//Show images while processing for debugging
		Size showsize;
		//showsize = Size(960, 540);
		showsize = Size(640, 480);


		bool foundAFull = false;
		bool foundBFull = false;

		//Use regular function for finding chessboard corners
		if (!useSBforfindingCorners) {

			cout << "Using findchessboardcorners " << endl;
			foundAFull = cv::findChessboardCorners(allImgsA[i], board_size, cornersA, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE); //CALIB_CB_ADAPTIVE_THRESH | | CALIB_CB_NORMALIZE_IMAGE
			foundBFull = cv::findChessboardCorners(allImgsB[i], board_size, cornersB, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
		}

		//Use Sector Based function
		else {
			cout << "Using findchessboardcornersSB (Sector Based) " << endl;
			foundAFull = cv::findChessboardCornersSB(allImgsA[i], board_size, cornersA, CALIB_CB_ACCURACY | CALIB_CB_ADAPTIVE_THRESH); //CALIB_CB_ADAPTIVE_THRESH | | CALIB_CB_NORMALIZE_IMAGE
			foundBFull = cv::findChessboardCornersSB(allImgsB[i], board_size, cornersB, CALIB_CB_ACCURACY | CALIB_CB_ADAPTIVE_THRESH);
		}
		if (!foundAFull || !foundBFull) //skip if we dont get a chessboard, extra check
		{
			cout << "error on " << i << "   no chessboard found.   found a and b   " << foundAFull << foundBFull << endl;
			//drawChessboardCorners(imageCopyLowResA, board_size, cornersA, foundA);
		}
		else
		{
			cout << "Found Board  " << i << endl;
			Mat grayA, grayB;
			/* no Cornersubpix if using findchessboardcornersSB */
			if (!useSBforfindingCorners) {
				cvtColor(allImgsA[i], grayA, COLOR_BGR2GRAY);
				cvtColor(allImgsB[i], grayB, COLOR_BGR2GRAY);

				cornerSubPix(grayA, cornersA, cv::Size(11, 11), cv::Size(-1, -1), TermCriteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.1));
				cornerSubPix(grayB, cornersB, cv::Size(11, 11), cv::Size(-1, -1), TermCriteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.1));
			}

			//Create the Chessboard object
			vector< Point3f > obj;
			for (int i = 0; i < squaresY; i++)
				for (int j = 0; j < squaresX; j++)
					obj.push_back(Point3f((float)j * squareLength, (float)i * squareLength, 0));


			if (downsampleChess) {
				// scale the corners back up
				for (int c = 0; c < cornersA.size(); c++) {
					cornersA[c].x *= 4;
					cornersA[c].y *= 4;
				}
				for (int c = 0; c < cornersB.size(); c++) {
					cornersB[c].x *= 4;
					cornersB[c].y *= 4;
				}
			}

			//Save all the points of this frame
			imagePointsA.push_back(cornersA);
			objectPointsA.push_back(obj);

			imagePointsB.push_back(cornersB);
			objectPointsB.push_back(obj);

			cout << " Success processed frame  " << i << endl;
		}
	}

	if (downsampleChess) { //Scale the images all back up just in case
		for (int i = 0; i < allImgsA.size(); i++) {
			resize(allImgsA[i], allImgsA[i], Size(), 4, 4);
			resize(allImgsB[i], allImgsB[i], Size(), 4, 4);
		}
	}


	/*
	...........CALIBRATION TIME....................
	*/

	//Calibrate each camera based on the detected points
	Mat cameraMatrixA, distCoeffsA;
	vector< Mat > rvecsA, tvecsA;
	double repErrorA;

	Mat cameraMatrixB, distCoeffsB;
	vector< Mat > rvecsB, tvecsB;
	double repErrorB;

	if (calibrationFlagsA & CALIB_FIX_ASPECT_RATIO) {
		cameraMatrixA = Mat::eye(3, 3, CV_64F);
		cameraMatrixA.at< double >(0, 0) = aspectRatio;
	}

	if (calibrationFlagsB & CALIB_FIX_ASPECT_RATIO) {
		cameraMatrixB = Mat::eye(3, 3, CV_64F);
		cameraMatrixB.at< double >(0, 0) = aspectRatio1;
	}


	int flag = 0;
	flag |= CALIB_FIX_K3; // Let's match what colmap does
	flag |= CALIB_FIX_K4;
	flag |= CALIB_FIX_K5;  //Sourish says ignore higher order distortion coeffs

	repErrorA = calibrateCamera(objectPointsA, imagePointsA, allImgsA[0].size(), cameraMatrixA, distCoeffsA, rvecsA, tvecsA, flag);
	cout << "Cam Matrix A:  " << cameraMatrixA << "  Calibration error Cam A: " << repErrorA << endl;

	repErrorB = calibrateCamera(objectPointsB, imagePointsB, allImgsB[0].size(), cameraMatrixB, distCoeffsB, rvecsB, tvecsB, flag);
	cout << "Cam Matrix B:  " << cameraMatrixB << "  Calibration error Cam B: " << repErrorB << endl;


	//Save Individual Camera Data
	bool saveOk = saveCameraParams(outputFolder + "/" + "_CamA.yml", imgSizeA, aspectRatio, calibrationFlagsA,
		cameraMatrixA, distCoeffsA, repErrorA);

	bool saveOk1 = saveCameraParams(outputFolder + "/" + "_CamB.yml", imgSizeB, aspectRatio1, calibrationFlagsB,
		cameraMatrixB, distCoeffsB, repErrorB);

	if (!saveOk) {
		cerr << "Cannot save output file CAMA" << endl;
		return 0;
	}

	if (!saveOk1) {
		cerr << "Cannot save output file CAMB" << endl;
		return 0;
	}

	cout << "CamA Rep Error: " << repErrorA << endl;
	cout << "CamA Calibration saved to " << outputFolder + "_CamA.yml" << endl;

	cout << "CamB Rep Error: " << repErrorB << endl;
	cout << "CamB Calibration saved to " << outputFolder + "_CamB.yml" << endl;

	/*
	.........STEREO CALIBRATION
		//Perform the Stereo Calibration between the Cameras
	*/
	cout << "Starting STEREO CALIBRATION Steps " << endl;


	Mat R, T, E, F; // Stereo Params

	//vector< vector< Point3f > > object_points;
	vector< Point2f > corners1, corners2;
	vector< vector< Point2f > > left_img_points, right_img_points;
	vector< Point3f > obj;

	//NOTE: stereo calibrate will change your CamMat and DisCoeffs
	double rms = stereoCalibrate(objectPointsA, imagePointsA, imagePointsB,
		cameraMatrixA, distCoeffsA,
		cameraMatrixB, distCoeffsB,
		imgSizeA, R, T, E, F, CALIB_FIX_INTRINSIC);

	// If the intrinsic parameters can be estimated with high accuracy for each of the cameras individually (for example, using calibrateCamera ), you are recommended to do so and then pass CALIB_FIX_INTRINSIC flag to the function along with the computed intrinsic parameters. 
	//Otherwise, if all the parameters are estimated at once, it makes sense to restrict some parameters, for example, pass CALIB_SAME_FOCAL_LENGTH and CALIB_ZERO_TANGENT_DIST flags, which is usually a reasonable assumption.
	cout << "camMatA after Stereo " << cameraMatrixA << endl;
	cout << "camMatB after Stereo " << cameraMatrixB << endl;


	//SAVE ALL THE STEREO DATA
	cout << "Stereo Calibration done with RMS error=" << rms << endl;
	cout << "Saving Stereo Calibration Files" << endl;
	// save intrinsic parameters
	FileStorage fs(outputFolder + "/" + "Stereo_intrinsics.yml", FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs << "M1" << cameraMatrixA << "D1" << distCoeffsA <<
			"M2" << cameraMatrixB << "D2" << distCoeffsB;
		fs.release();
	}
	else
		cout << "Error: can not save the intrinsic parameters\n";


	fs.open(outputFolder + "/" + "Stereo_extrinsics_preRect.yml", FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs << "R" << R << "T" << T;
		fs.release();
	}
	else
		cout << "Error: can not save the extrinsic parameters\n";


	//Perform Stereo Rectification

	printf("Starting Stereo Rectification\n");
	Mat R1, R2, P1, P2, Q;
	Rect validRoi[2];

	stereoRectify(cameraMatrixA, distCoeffsA,
		cameraMatrixB, distCoeffsB,
		imgSizeA, R, T, R1, R2, P1, P2, Q,
		CALIB_ZERO_DISPARITY,
		1, imgSizeA, &validRoi[0], &validRoi[1]);
	// -1 will give default scaling
	//0 means no black pixels in image
	//1 means all valid pixels are in image (black usually around borders)

	fs.open(outputFolder + "/" + "Stereo_extrinsics.yml", FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
		fs.release();
	}
	else
		cout << "Error: can not save the extrinsic parameters\n";

	//This is the main file we want to get out of this program for the Structured Light Decoding
	bool saveOkStereo = saveCameraParamsStereo(outputFolder + "/" + "stereoCalibrationParameters_camAcamB.yml", imgSizeA, imgSizeB, aspectRatio, aspectRatio1, calibrationFlagsA, calibrationFlagsB,
		cameraMatrixA, cameraMatrixB, distCoeffsA, distCoeffsB, repErrorA, repErrorB, R, T, R1, R2, P1, P2, Q, rms, inputVideoA.get(CAP_PROP_EXPOSURE), inputVideoB.get(CAP_PROP_EXPOSURE), inputVideoA.get(CAP_PROP_FOCUS), inputVideoB.get(CAP_PROP_FOCUS), squaresX, squaresY, squareLength);

	printf("Done Stereo Rectification\n");
	waitKey();
	destroyAllWindows();
	printf("Finished All Tasksn\n");

	return 0;
}
