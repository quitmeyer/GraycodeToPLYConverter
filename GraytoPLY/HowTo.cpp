/*  Use example to export 3D point cloud from reprojectImageTo3D (OpenCV) to
.ply file format (ascii, binary big endian or binary little endian) as vertices */

// Minimal OpenCV dependancies
#include "DataExporter.hpp"
#include "opencv2/core/core.hpp"

// Variables (must come from your code)
cv::Mat coords3d; //from reprojectImageTo3D

// Constructor for class
// first argument: Matrix of (x,y,z) coordinates (3 channels)
// second argument: Source image (color, 3 channels BGR) used to associate color to each vertex
// third argument: Output file name and path
// fourth argument: File format (described in .ply format description), see header file
//                  FileFormat enum for choices (PLY_ASCII, PLY_BIN_BIGEND, PLY_BIN_LITEND)
//DataExporter data(coords3d, img, "outputfile.ply", FileFormat::PLY_BIN_BIGEND);
//data.exportToFile();