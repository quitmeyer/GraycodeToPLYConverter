/*
Graycode Point Cloud Generator

This program takes a series of graycode structured light scans from a pair of 2 cameras and a projector

give it
-a data file that includes the list of images from the cameras
-calibration data from the cameras

and it will

-Rectify the images
-Decode the Graycode pattern in the images

*/


#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/structured_light.hpp>
#include <opencv2/opencv_modules.hpp>
#include <fstream>  
#include "DataExporter.hpp"

#include <iomanip>

using namespace std;
using namespace cv;


static const char* keys =
{

	"{images_list          |C:/Users/andre/Desktop/Glowcake Hoss/Calibrations/TestDecode/SLdata.yml|}" //For windows have to change all backslashes to forward slashes
	"{calib_param_path     |C:/Users/andre/Desktop/Glowcake Hoss/Calibrations/TestDecode/stereoCalibrationParameters_camAcamB.yml| Calibration_parameters            }"
	"{exportPLY     |false| spend time creating a PLY image}"
	"{@white_thresh     |<none>| The white threshold height (optional)}"
	"{@black_thresh     |<none>| The black threshold (optional)}" };

static void help()
{
	cout << "\nThis example shows how to use the \"Structured Light module\" to decode a previously acquired gray code pattern, generating a pointcloud"
		"\nCall:\n"
		"./example_structured_light_pointcloud <proj_width> <proj_height> <images_list> <calib_param_path>  <white_thresh> <black_thresh>\n"
		<< endl;
}

static bool readStringListCameraImages(const string& filename, vector<string>& l)
{
	l.resize(0);
	FileStorage fs(filename, FileStorage::READ);
	if (!fs.isOpened())
	{
		cerr << "failed to open " << filename << endl;
		return false;
	}
	FileNode n = fs.getFirstTopLevelNode();
	n = fs["camA"];
	if (n.type() != FileNode::SEQ)
	{
		cerr << "cam A images are not a sequence! FAIL" << endl;
		return false;
	}

	FileNodeIterator it = n.begin(), it_end = n.end();
	for (; it != it_end; ++it)
	{
		l.push_back((string)*it);
	}

	n = fs["camB"];
	if (n.type() != FileNode::SEQ)
	{
		cerr << "cam B images are not a sequence! FAIL" << endl;
		return false;
	}

	it = n.begin(), it_end = n.end();
	for (; it != it_end; ++it)
	{
		l.push_back((string)*it);
	}

	if (l.size() % 2 != 0)
	{
		cout << "Error: the image list contains odd (non-even) number of elements\n";
		return false;
	}
	return true;
}

int main(int argc, char** argv)
{
	structured_light::GrayCodePattern::Params params;
	CommandLineParser parser(argc, argv, keys);

	String images_file = parser.get<String>("images_list", true);
	String calib_file = parser.get<String>("calib_param_path", true);
	bool exportPLY = parser.get<bool>("exportPLY", true);


	string outputFolder = "C:/Users/andre/Desktop/Glowcake Hoss/Calibrations/TestDecode/data";

	vector<string> imagelist;
	bool ok = readStringListCameraImages(images_file, imagelist);
	if (!ok || imagelist.empty())
	{
		cout << "can not open " << images_file << " or the string list is empty" << endl;
		help();
		return -1;
	}

	FileStorage fsI(images_file, FileStorage::READ);
	if (!fsI.isOpened())
	{
		cout << "Failed to open Image List Data File." << endl;
		help();
		return -1;
	}
	//Read Data from Image list Projector File

	vector<string> projH;
	vector<string> projW;
	vector<string> diagFOV;

	//Get projector details from the images list 
	//FYI the projectors we tend to use at GLOWCAKE are either   1366 768  or 1920 1080
	FileNode n = fsI.getFirstTopLevelNode();

	n = fsI["projW"];
	FileNodeIterator it = n.begin(), it_end = n.end();

	it = n.begin(), it_end = n.end();
	for (; it != it_end; ++it)
	{
		projW.push_back((string)*it);
	}

	n = fsI["projH"];
	it = n.begin(), it_end = n.end();

	for (; it != it_end; ++it)
	{
		projH.push_back((string)*it);
	}

	n = fsI["projFOV"];
	it = n.begin(), it_end = n.end();

	for (; it != it_end; ++it)
	{
		diagFOV.push_back((string)*it);
	}

	cout << "projector Width " << projW[0] << endl;

	cout << "projector Height  " << projH[0] << endl;

	cout << "Proj Diagonal FOV  " << diagFOV[0] << endl;
	cout << "Create PLY?  " << exportPLY << endl;


	params.width = stoi(projW[0]);
	params.height = stoi(projH[0]);
	double dFOV = ::atof(diagFOV[0].c_str());

	if (images_file.empty() || calib_file.empty() || params.width < 1 || params.height < 1)
	{
		help();
		return -1;
	}

	// Set up GraycodePattern with params
	Ptr<structured_light::GrayCodePattern> graycode = structured_light::GrayCodePattern::create(params);
	size_t white_thresh = 0;
	size_t black_thresh = 0;

	//TODO update this so it s not count based
	if (argc == 7)
	{
		// If passed, setting the white and black threshold, otherwise using default values
		white_thresh = parser.get<unsigned>(4);
		black_thresh = parser.get<unsigned>(5);

		graycode->setWhiteThreshold(white_thresh);
		graycode->setBlackThreshold(black_thresh);
	}

	FileStorage fs(calib_file, FileStorage::READ);
	if (!fs.isOpened())
	{
		cout << "Failed to open Calibration Data File." << endl;
		help();
		return -1;
	}

	// Loading calibration parameters
	Mat camAintrinsics, camAdistCoeffs, camBintrinsics, camBdistCoeffs, R, T;

	Mat R1, R2, P1, P2, Q;

	fs["camA_intrinsics"] >> camAintrinsics;
	fs["camB_intrinsics"] >> camBintrinsics;
	fs["camA_distorsion"] >> camAdistCoeffs;
	fs["camB_distorsion"] >> camBdistCoeffs;
	fs["R"] >> R;
	fs["T"] >> T;
	fs["R1"] >> R1;
	fs["R2"] >> R2;
	fs["P1"] >> P1;
	fs["P2"] >> P2;
	fs["Q"] >> Q;



	cout << "camAintrinsics" << endl << camAintrinsics << endl;
	cout << "camAdistCoeffs" << endl << camAdistCoeffs << endl;
	cout << "camBintrinsics" << endl << camBintrinsics << endl;
	cout << "camBdistCoeffs" << endl << camBdistCoeffs << endl;
	cout << "T" << endl << T << endl << "R" << endl << R << endl;
	cout << "R1" << endl << R1 << endl << "R2" << endl << R2 << endl;
	cout << "P1" << endl << P1 << endl << "P2" << endl << P2 << endl;
	cout << "Q" << endl << Q << endl;



	if ((!R.data) || (!T.data) || (!R1.data) || (!R2.data) || (!P1.data) || (!P2.data) || (!camAintrinsics.data) || (!camBintrinsics.data) || (!camAdistCoeffs.data) || (!camBdistCoeffs.data))
	{
		cout << "Failed to load cameras' calibration parameters" << endl;
		help();
		return -1;
	}

	size_t numberOfPatternImages = graycode->getNumberOfPatternImages();
	vector<vector<Mat> > captured_pattern;
	captured_pattern.resize(2);//Two cameras
	captured_pattern[0].resize(numberOfPatternImages);//CAM A
	captured_pattern[1].resize(numberOfPatternImages); //CAM B

	Mat color = imread(imagelist[numberOfPatternImages], IMREAD_COLOR); //This is the WHITE image taken from Camera A used as a COLOR REFERENCE for the scene
	Size imagesSize = color.size();

	cout << "Loading Data Complete. Beginning Processing..." << endl;

	/*
	.......PROCESSING STAGES.........
	*/


	//LOADING ALL IMAGES


	// Loading pattern images
	for (size_t i = 0; i < numberOfPatternImages; i++)
	{
		cout << i + 1 << " of " << numberOfPatternImages << endl;

		captured_pattern[0][i] = imread(imagelist[i], IMREAD_GRAYSCALE); //Read Camera A in as grayscale
		captured_pattern[1][i] = imread(imagelist[i + numberOfPatternImages + 2], IMREAD_GRAYSCALE); //Read Camera B in as grayscale

		//Check to make sure the pics actually loaded!
		if ((!captured_pattern[0][i].data) || (!captured_pattern[1][i].data))
		{
			cout << "Empty images at index " << i << " " << imagelist[i] << "  or  " << imagelist[i + numberOfPatternImages + 2] << endl;
			help();
			return -1;
		}

		
	}

	//get the white and black images
	vector<Mat> blackImages;
	vector<Mat> whiteImages;

	blackImages.resize(2);
	whiteImages.resize(2);

	// Loading images (all white + all black) needed for shadows computation
	cvtColor(color, whiteImages[0], COLOR_RGB2GRAY); // White image, Camera A

	whiteImages[1] = imread(imagelist[2 * numberOfPatternImages + 2], IMREAD_GRAYSCALE);
	blackImages[0] = imread(imagelist[numberOfPatternImages + 1], IMREAD_GRAYSCALE);
	blackImages[1] = imread(imagelist[2 * numberOfPatternImages + 2 + 1], IMREAD_GRAYSCALE);


	cout << "done loading images" << endl;




	//Decoding Projector Pixels	
	bool decodeVis = false;
	if (decodeVis) {
		cout << "Decoding the Projected Pixels camA" << endl;

		Mat camADecodedViz(whiteImages[0].rows, whiteImages[0].cols, CV_8UC3); // Start with all black images
		Mat camBDecodedViz(whiteImages[0].rows, whiteImages[0].cols, CV_8UC3);
		Point projPixelA; //= new Point(0.0, 0.0);

		//coordinates of projected pixels
	//Xc is camera pixel for cams A and B, Xp is projector location
		std::vector<int>	XcA;
		std::vector<int>	YcA;
		std::vector<int>	XcB;
		std::vector<int>	YcB;
		std::vector<int>	XpA;
		std::vector<int>	YpA;
		std::vector<int>	XpB;
		std::vector<int>	YpB;

		for (int i = 0; i < camADecodedViz.rows; i++)
		{
			for (int j = 0; j < camADecodedViz.cols; j++)
			{

				if (abs((float)whiteImages[0].at<uchar>(i, j) - (float)blackImages[0].at<uchar>(i, j)) > 30) {


					bool error = graycode->getProjPixel(captured_pattern[0], j, i, projPixelA); //Get pixel based on view of first camera

					if (error) {
						// cout << endl << " Error Pixel no pattern here  i" << i <<"  j "<<j<< endl;
						Vec3b color; //BGR
						color[0] = 0;
						color[1] = 0;
						color[2] = 255;
						camADecodedViz.at<Vec3b>(i, j) = color;
					}
					else { // Pattern  was sucessfully detected here


						//Decode the pattern 
						 /*
					Spit out image grayscale
					CSV - 4 cols, Xc, Yc, Xp, Yp,
					Rows - pixels of camera
					*/

						Vec3b color;
						color[0] = static_cast<double>(projPixelA.x) / params.width * 255.0;
						color[1] = static_cast<double>(projPixelA.y) / params.height * 255.0;
						color[2] = 0;

						camADecodedViz.at<Vec3b>(i, j) = color;


						XcA.push_back(j);
						YcA.push_back(i);
						XpA.push_back(projPixelA.y);
						YpA.push_back(projPixelA.y);
					}
				}
				else {
					Vec3b color;
					color[0] = 0;
					color[1] = 0;
					color[2] = 0;

					camADecodedViz.at<Vec3b>(i, j) = color;
				}
			}

		}
		imwrite(outputFolder + "/" + "Graycode Decode camA" + ".png", camADecodedViz);

	}



	// Stereo rectify IMAGES

	bool rectify = true;
	if (rectify) {
		cout << "Rectifying images..." << endl;
		/*
		Rect validRoi[2];
		//cout << "R before Stereorectify" << R << "  T before "<<T<< endl;


		stereoRectify(camAintrinsics, camAdistCoeffs, camBintrinsics, camBdistCoeffs, imagesSize, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY,
			-1, imagesSize, &validRoi[0], &validRoi[1]);

		cout << "R After Stereorectify" << R << "  T after " << T << endl;


		//StereoRectify NOTE! Operation flags that may be zero or CV_CALIB_ZERO_DISPARITY. If the flag is set, the function makes the principal points of each camera have the same pixel coordinates in the rectified views. And if the flag is not set, the function may still shift the images in the horizontal or vertical direction (depending on the orientation of epipolar lines) to maximize the useful image area.
		*/
		Mat map1x, map1y, map2x, map2y;
		initUndistortRectifyMap(camAintrinsics, camAdistCoeffs, R1, P1, imagesSize, CV_32FC1, map1x, map1y);
		initUndistortRectifyMap(camBintrinsics, camBdistCoeffs, R2, P2, imagesSize, CV_32FC1, map2x, map2y);

		namedWindow("Show Rectified Images", WINDOW_NORMAL);

		resizeWindow("Show Rectified Images", 600, 400);

		// Loading pattern images
		for (size_t i = 0; i < numberOfPatternImages; i++)
		{
			cout << i + 1 << " of " << numberOfPatternImages << endl;

			//Recitify the images from both cameras
			remap(captured_pattern[0][i], captured_pattern[0][i], map1x, map1y, INTER_NEAREST, BORDER_CONSTANT, Scalar());
			remap(captured_pattern[1][i], captured_pattern[1][i], map2x, map2y, INTER_NEAREST, BORDER_CONSTANT, Scalar());

			imshow("Show Rectified Images", captured_pattern[0][i]); // show Cam A undistorted
			waitKey(1);
		}



		remap(color, color, map2x, map2y, INTER_NEAREST, BORDER_CONSTANT, Scalar()); //Rectify the color image Reference

		remap(whiteImages[0], whiteImages[0], map1x, map1y, INTER_NEAREST, BORDER_CONSTANT, Scalar());
		remap(whiteImages[1], whiteImages[1], map2x, map2y, INTER_NEAREST, BORDER_CONSTANT, Scalar());

		//For debugging comparisons. let's save a copy of these white images from both cameras
		imwrite(outputFolder + "/" + "whiteimg rect camA" + ".png", whiteImages[0]);
		imwrite(outputFolder + "/" + "whiteimg rect camB" + ".png", whiteImages[1]);


		remap(blackImages[0], blackImages[0], map1x, map1y, INTER_NEAREST, BORDER_CONSTANT, Scalar());
		remap(blackImages[1], blackImages[1], map2x, map2y, INTER_NEAREST, BORDER_CONSTANT, Scalar());

		cout << "done" << endl;

	}

	

	cout << endl << "Decoding Graycode pattern ..." << endl;
	Mat disparityMap;

	//Function built into the opencv Structured light Contrib module. Takes all the stacks of graycoded images
	//It then creates a disparity map
	bool decoded = graycode->decode(captured_pattern, disparityMap, blackImages, whiteImages,
		structured_light::DECODE_3D_UNDERWORLD);

	if (decoded)
	{
		cout << endl << "pattern decoded! Hooray!" << endl;

		//TODO put the function from the scanning code in here that creates our CSV of matches between pixels and points on the decoded imagery

		// To better visualize the result, apply a colormap to the computed disparity
		double min;
		double max;
		minMaxIdx(disparityMap, &min, &max);
		Mat cm_disp, scaledDisparityMap;
		cout << "disp min " << min << endl << "disp max " << max << endl;
		max = 284;
		min = -312; // these are arbitrary numbers that seem to be more around the actual scale
		convertScaleAbs(disparityMap, scaledDisparityMap, 255 / (max - min));// TODO for some reason our color map shoudl look rainbow like, but is just barely visiable as blue. our scaling must be wrong
		//disparityMap.copyT312o(scaledDisparityMap);
		//scaledDisparityMap.convertTo(scaledDisparityMap, CV_8UC1);
		applyColorMap(scaledDisparityMap, cm_disp, COLORMAP_RAINBOW);

		//Shows the disparity map as a RECTIFIED image
		imshow("cm disparity m", cm_disp);
		waitKey(1);
		imwrite(outputFolder + "/" + "cm disparity m" + ".png", cm_disp);



		// Compute the point cloud
		Mat pointCloud;
		disparityMap.convertTo(disparityMap, CV_32FC1); // Change the MAT type so it plays well with reprojectImageto3D function

		reprojectImageTo3D(disparityMap, pointCloud, Q, false, -1); // Takes the disparity map along with Q, a value we get from Stereo Rectify that turns Z coordinates into true distances
		//Since handleMissingValues=true, 
		//then pixels with the minimal disparity that corresponds to the outliers (see StereoBM::operator() ) are transformed to 3D points with a very large Z value (currently set to 10000).

		cout << endl << "  Image Reprojected to 3D  " << endl;

		// maybe perspectiveTransform is what is necessary to reproject the projector?


		// Compute a mask to remove background
		Mat dst, thresholded_disp;
		threshold(scaledDisparityMap, thresholded_disp, 0, 255, THRESH_OTSU + THRESH_BINARY); // OTSU calculates some kind of optimal threshold, then binary just sends up to black or white
		resize(thresholded_disp, dst, Size(640 * 2, 480 * 2), 0, 0, INTER_LINEAR_EXACT);
		//TODO make sure the mask we created  via the Scaled Disparity Map is good


		imshow("threshold disp otsu", dst);
		waitKey(1);
		// Apply the mask to the point cloud
		Mat pointcloud_tresh, color_tresh;

		pointCloud.copyTo(pointcloud_tresh, thresholded_disp); // This applies a thresholded mask on the pointcloud, but honestly i don't see a difference
		//pointCloud.copyTo(pointcloud_tresh);
		color.copyTo(color_tresh, thresholded_disp);

		//can I save the pointcloud mat as an Image? Yes you can!
		imwrite(outputFolder + "/" + "Raw Pointcloud Save" + ".png", pointcloud_tresh);
		imwrite(outputFolder + "/" + "WB mask thresholded disp " + ".png", thresholded_disp);


		/*
		Export and Visualize Data
		*/


		//Export the MAT as a PLY for 3D viewing
		if (exportPLY) {
			cout << endl << "  Starting to Export the PLY  " << endl;
			DataExporter data(pointcloud_tresh, color_tresh, "C:/Users/andre/Desktop/Glowcake Hoss/Calibrations/TestDecode/data/pointcloud.ply", FileFormat::PLY_ASCII);
			//        DataExporter data(pointCloud, color, "C:/Users/andre/Desktop/Glowcake Hoss/Calibrations/TestDecode/data/pointcloud.ply", FileFormat::PLY_BIN_LITEND);

			data.exportToFile();

			cout << endl << "  Finished Exporting PLY  " << endl;
		}


		/*		Reprojections
		//Try to reproject  Canonical Camera's points to examine calibration
		*/


		//Canonical Camera
		bool calcCanonCamA = true;
		if (calcCanonCamA) {
			cout << endl << " Calculating Canon Camera Stuff " << endl;

			cout << "Camera Matrix Cam A " << "\n" << camAintrinsics << endl;

			//Size imageSizeCanon = Size(pointcloud_tresh.cols, pointcloud_tresh.rows);

			Point projPixelA; //= new Point(0.0, 0.0);

			std::vector<std::vector<cv::Vec2f>> processedImagePointsP;
			vector<cv::Vec2f> imagePoints;
			cv::Vec2f apoint;

			std::vector<std::vector<cv::Point3f>> processedobjectPointsP;
			std::vector<cv::Point3f> objectPointsP;

			double minIDX;
			double maxIDX;
			minMaxIdx(pointcloud_tresh, &minIDX, &maxIDX);
			cout << "minMaxIDX " << "\n" << minIDX << "   max= " << maxIDX << endl;


			float minDepth = FLT_MAX;
			float maxDepth = -FLT_MAX;
			float depth = 9;
			//const float* input = pointcloud_tresh.ptr<float>(0);






			//Loop through the pointcloud to gather corresponding pairs between
			//-all the image points (x,y)
			//	-All the detected object points (X,Y,Z) (I think the XY are different than the x,y of the pixels
			//According to reprojectImageto3D  Each element of _3dImage(x,y) contains 3D coordinates of the point (x,y) computed from the disparity map.
			//Remember Points and Size go (x,y); (width,height) ,- Mat has ( row, col).

			cout << "Starting Pointcloud Loop" << endl;

			for (int i = 0; i < pointcloud_tresh.rows; i++)
			{
				for (int j = 0; j < pointcloud_tresh.cols; j++)
				{

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


						//The pointcloud stores a set of 3D points. Every point in the Mat (i,j) refers to a 3D point at 0 , 1, 2 (aka xyz)
					float x = pointcloud_tresh.at<Vec3f>(i, j)[0];
					float y = pointcloud_tresh.at<Vec3f>(i, j)[1];
					float z = pointcloud_tresh.at<Vec3f>(i, j)[2];
					//  cout << endl << " almost Good Pixel at  i " << i <<"  j "<<j<< "  x " << x << "  y " << y << "  z " << z << endl;



					depth = sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2));
					//Calculate min and max depth for the loop
					if (depth > maxDepth && depth != 10000) maxDepth = depth;
					if (depth < minDepth && depth != 10000) minDepth = depth;


					if (x == 0 && y == 0 && z == 0) { // skip empty points
					}
					else {
						//Get the Image Pixel points
						cv::Vec2f apoint;
						imagePoints.push_back(cv::Vec2f(projPixelA.x, projPixelA.y));
						objectPointsP.push_back(cv::Point3f(x, y, z));
					}

				} // pointcloud cols
			}//pointcloud rows



			cout << endl << "min depth  " << minDepth << "   maxDepth " << maxDepth << endl;


			processedImagePointsP.push_back(imagePoints); // Do we ever really need the image points past here, i don't think so
			processedobjectPointsP.push_back(objectPointsP);

			Mat rvecs0 = Mat::eye(3, 3, CV_64F);

			Mat tvecs0 = Mat::zeros(3, 1, CV_64F);
			cout << endl << " Prep reproject into canon cam  " << "\n" << "Camera Matrix Canon " << "\n" << camAintrinsics << "\n" << "  rvecs " << "\n" << rvecs0 << "\n" << "  Tvecs " << "\n" << tvecs0 << endl;

			//Try to project the points and see what we see from canonical camera (camera A)'s POV
			std::vector<cv::Point2f> CanonImagePoints;
			std::vector<cv::Point2f> CamBImagePoints;

			//Mat projimagePoints2;

			//Project Through Camera A (Canonical)
			projectPoints(processedobjectPointsP.front(), rvecs0, tvecs0, camAintrinsics, camAdistCoeffs, CanonImagePoints);// 

			Mat Rt;
				transpose(R, Rt);
			//Project Through Camera B
			projectPoints(processedobjectPointsP.front(), Rt, -T, camBintrinsics, camBdistCoeffs, CamBImagePoints); // Cam B is quite close when it is R and -T


			// visualize the reprojection Canon Cam A
			Mat projIM(whiteImages[0].rows, whiteImages[0].cols, CV_8UC3, Scalar(10, 10, 40));
			for (int i = 0; i < CanonImagePoints.size(); i++) {


				int x = CanonImagePoints[i].x; // note this is rounding the values we are actually getting into integer pixel values
				int y = CanonImagePoints[i].y;
				int z = objectPointsP[i].z;


				Vec3b color; // RGB
				color[0] = sqrt(x ^ 2 + y ^ 2 + z ^ 2) / ((float)maxDepth - minDepth)* 255.0;
				color[1] = 255;
				color[2] = 0;

				if (x > 0 && x < projIM.cols && y>0 && y < projIM.rows) {
					projIM.at<Vec3b>(y, x) = color; // mats are always ROW, COL,
				}
			}
			namedWindow("Project Points Canon Cam image", WINDOW_NORMAL);
			resizeWindow("Project Points Canon Cam image", 800, 600);
			imshow("Project Points Canon Cam image", projIM);
			waitKey(1);
			imwrite(outputFolder + "/" + "Canon Cam Image Reprojection" + ".png", projIM);

			// visualize the reprojection through Cam B
			Mat projIMB(whiteImages[0].rows, whiteImages[0].cols, CV_8UC3, Scalar(10, 10, 40));
			for (int i = 0; i < CamBImagePoints.size(); i++) {


				int x = CamBImagePoints[i].x; // note this is rounding the values we are actually getting into integer pixel values
				int y = CamBImagePoints[i].y;
				int z = objectPointsP[i].z;


				Vec3b color;
				color[0] = sqrt(x ^ 2 + y ^ 2 + z ^ 2) * 255.0 / (maxIDX - minIDX);
				color[1] = 255;
				color[2] = 0;

				if (x > 0 && x < projIMB.cols && y>0 && y < projIMB.rows) {
					projIMB.at<Vec3b>(y, x) = color; // mats are always ROW, COL,
				}
			}
			namedWindow("Project Points CAMB", WINDOW_NORMAL);
			resizeWindow("Project Points CAMB", 800, 600);
			imshow("Project Points CAMB", projIMB);
			waitKey(1);
			imwrite(outputFolder + "/" + "CAMB reproject Image" + ".png", projIMB);


			//Calc Projector
			cout << endl << " Calculating projector position " << endl;

			Mat distCoeffsP;
			float aspectRatio = 16 / 9.0;

			//Set intial guess for camera matrix
			Mat cameraMatrixG;

			cameraMatrixG = Mat::eye(3, 3, CV_32FC1);
			float w = params.width; // projector w
			float h = params.height; // projector h
			float diagnonalFOV = dFOV;

			///Building a guess from our measured fields of view of the projectors

			//1920 x 1080  big projector 35.1	    0.6126105675
		   // 1366 x 768 small projector 72.88	1.271995959

			float d = sqrt(powf(w, 2) + powf(h, 2));
			float f = (d / 2) * cos(diagnonalFOV / 2) / sin(diagnonalFOV / 2);  // old guess  1.732; // 1.732 = cotangent(1.0472/2) where 1.0472 is 60 degrees in radians)

			cameraMatrixG.at< float >(0, 0) = f;
			cameraMatrixG.at< float >(1, 1) = f;
			cameraMatrixG.at< float >(0, 2) = w / 2; // assume it's about in the center
			cameraMatrixG.at< float >(1, 2) = h / 2; // assume it's about in the center

			cout << endl << " Initial Guess at Projector Camera Matrix Intrinsics " << "\n" << "Projector Camera Matrix " << "\n" << cameraMatrixG << endl;

			vector<Mat> rvecsP, tvecsP;
			// Size imageSizeP = Size(params.width, params.height);

			Size imageSizeP = Size(pointcloud_tresh.cols, pointcloud_tresh.rows);

			//Run calibrate camera to try to get the intrinsics and extrinsics of the projector
			calibrateCamera(processedobjectPointsP, processedImagePointsP, imageSizeP, cameraMatrixG, distCoeffsP, rvecsP, tvecsP, CALIB_USE_INTRINSIC_GUESS);



			cout << "Projector Camera Matrix after calibration " << "\n" << cameraMatrixG << "\n" << "  rvecs " << "\n" << rvecsP.front() << "\n" << "  Tvecs " << "\n" << tvecsP.front() << endl;

			//Try to project the points and see what we see from the projector's point of view
			std::vector<cv::Point2f> projimagePoints2;
			//Mat projimagePoints2;
			projectPoints(processedobjectPointsP.front(), rvecsP.front(), tvecsP.front(), cameraMatrixG, distCoeffsP, projimagePoints2);


			Mat projIMprojector(blackImages[0].rows, blackImages[0].cols, CV_8UC3, Scalar(10, 10, 40));
			for (int i = 0; i < projimagePoints2.size(); i++) {


				int x = projimagePoints2[i].x; // note this is rounding the values we are actually getting into integer pixel values
				int y = projimagePoints2[i].y;


				Vec3b color;
				color[0] = 110;
				color[1] = 20;
				color[2] = 255;

				if (x > 0 && x < projIMprojector.cols && y>0 && y < projIMprojector.rows) {
					projIMprojector.at<Vec3b>(y, x) = color; // mats are always ROW, COL,
				}


			}
			namedWindow("Project Points Projector image", WINDOW_NORMAL);
			resizeWindow("Project Points Projector image", 600, 600);
			imshow("Project Points Projector image", projIMprojector);


		}


		/*		//Try to calculate PROJECTOR's Intrinsic and Extrinsic matrix using the calibrate camera function in reverse */

		bool calcProj = false;

		if (calcProj) {
			cout << endl << " Calculating projector position " << endl;

			Mat  distCoeffsP;
			float aspectRatio = 16 / 9.0;


			//Set intial guess for camera matrix

			Mat cameraMatrixG;
			cameraMatrixG = Mat::eye(3, 3, CV_32FC1);
			float w = params.width; // projector w
			float h = params.height; // projector h
			float diagnonalFOV = dFOV;


			///Building a guess from our measured fields of view of the projectors

			//1920 x 1080  big projector 35.1	    0.6126105675
		   // 1366 x 768 small projector 72.88	1.271995959


			float d = sqrt(powf(w, 2) + powf(h, 2));
			float f = (d / 2) * cos(diagnonalFOV / 2) / sin(diagnonalFOV / 2);  // old guess  1.732; // 1.732 = cotangent(1.0472/2) where 1.0472 is 60 degrees in radians)

			cameraMatrixG.at< float >(0, 0) = f;
			cameraMatrixG.at< float >(1, 1) = f;
			cameraMatrixG.at< float >(0, 2) = w / 2; // assume it's about in the center
			cameraMatrixG.at< float >(1, 2) = h / 2; // assume it's about in the center

			cout << endl << " Initial Guess at Camera Matrix Intrinsics " << "\n" << "Camera Matrix " << "\n" << cameraMatrixG << endl;


			vector<Mat> rvecsP, tvecsP;
			double repError1;
			// Size imageSizeP = Size(params.width, params.height);

			Size imageSizeP = Size(pointcloud_tresh.cols, pointcloud_tresh.rows);

			Point projPixelA; //= new Point(0.0, 0.0);

			std::vector<std::vector<cv::Vec2f>> processedImagePointsP;
			vector<cv::Vec2f> imagePoints;
			cv::Vec2f apoint;

			std::vector<std::vector<cv::Point3f>> processedobjectPointsP;
			std::vector<cv::Point3f> objectPointsP;





			// const float* pData = pointcloud_tresh.ptr<float>(0);
			 //float* input = (float*)(pointcloud_tresh.data);
			const float* input = pointcloud_tresh.ptr<float>(0);
			//Remember Points and Size go (x,y); (width,height) ,- Mat has (row,col).
			for (int i = 0; i < pointcloud_tresh.rows; i++)
			{
				for (int j = 0; j < pointcloud_tresh.cols; j++)
				{
					bool error = graycode->getProjPixel(captured_pattern[0], j, i, projPixelA); //Get pixel based on view of first camera
					if (error) {
						// cout << endl << " Error Pixel no pattern here  i" << i <<"  j "<<j<< endl;

					}
					else // Pattern  was sucessfully detected here
					{

						// float x = input[pointcloud_tresh.step * j + i];
						 //float y = input[pointcloud_tresh.step * j + i + 1];
						 //float z = input[pointcloud_tresh.step * j + i + 2];



						float x = pointcloud_tresh.at<Vec3f>(i, j)[0];
						float y = pointcloud_tresh.at<Vec3f>(i, j)[1];
						float z = pointcloud_tresh.at<Vec3f>(i, j)[2];
						//  cout << endl << " almost Good Pixel at  i " << i <<"  j "<<j<< "  x " << x << "  y " << y << "  z " << z << endl;

						if (x == 0 && y == 0 && z == 0) { // skip empty points
						}
						else {
							//? Any sucess?
						   // cout << endl << " Good Pixel at  i" << i << "  j " << j << "  x " << x << "  y " << y << "  z " << z << endl;


							//Get the Image Pixel points
							cv::Vec2f apoint;
							imagePoints.push_back(cv::Vec2f(projPixelA.x, projPixelA.y));



							//pointcloud_tresh.at<cv::Vec3b>(i, j)[0];

						   // img.at<cv::Vec3b>(i, j)[0]


							//unsigned char b = input[img.step * j + i];
							//unsigned char g = input[img.step * j + i + 1];
							//unsigned char r = input[img.step * j + i + 2];

							//get the point cloud loaded
							//objectPointsP.push_back(pointcloud_tresh.at<Point3f>(i, j));

							objectPointsP.push_back(cv::Point3f(x, y, z));
						}
					}
				}
			}

			processedImagePointsP.push_back(imagePoints);
			processedobjectPointsP.push_back(objectPointsP);
			// Mat processedImagePointsP;
			// disparityMap.copyTo(processedImagePointsP);

			calibrateCamera(processedobjectPointsP, processedImagePointsP, imageSizeP, cameraMatrixG, distCoeffsP, rvecsP, tvecsP, CALIB_USE_INTRINSIC_GUESS);
			//       calibrateCamera(processedobjectPointsP, processedImagePointsP, imageSizeP, cameraMatrixP, distCoeffsP, rvecsP, tvecsP);


			Mat rvecs0 = rvecsP[0];
			Mat tvecs0 = tvecsP[0];
			cout << endl << " Calibrate Camera Points  " << "\n" << "Camera Matrix " << "\n" << cameraMatrixG << "\n" << "  rvecs " << "\n" << rvecs0 << "\n" << "  Tvecs " << "\n" << tvecs0 << endl;

			//Try to project the points and see what we see from the projector's point of view
			std::vector<cv::Point2f> projimagePoints2;
			//Mat projimagePoints2;
			projectPoints(processedobjectPointsP.front(), rvecsP.front(), tvecsP.front(), cameraMatrixG, distCoeffsP, projimagePoints2);
			//cout << endl << " projImagepoints!  " << endl << projimagePoints2 << endl;

			//Visualize the projection
			//Against the original?

			//Mat projIM;
		   // disparityMap.copyTo(projIM); // 32fc1

			Mat projIM(blackImages[0].rows, blackImages[0].cols, CV_8UC3, Scalar(10, 10, 40));
			for (int i = 0; i < projimagePoints2.size(); i++) {


				int x = projimagePoints2[i].x; // note this is rounding the values we are actually getting into integer pixel values
				int y = projimagePoints2[i].y;


				Vec3b color;
				color[0] = 110;
				color[1] = 255;
				color[2] = 0;

				if (x > 0 && x < projIM.cols && y>0 && y < projIM.rows) {
					projIM.at<Vec3b>(y, x) = color; // mats are always ROW, COL,
				}


			}
			namedWindow("Project Points Projector image", WINDOW_NORMAL);
			resizeWindow("Project Points Projector image", 600, 600);
			imshow("Project Points Projector image", projIM);
		}
	}

	cout << endl << " All  Finished  " << endl;



	waitKey();
	return 0;

}
