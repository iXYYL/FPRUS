/*
* Author: Corfox
* Date: 2015.10.22
*/

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <string>
#include <iostream>
#include <fstream>
#include "transformation.h"

using std::string;
using std::ifstream;
using std::cout;
using std::endl;
using cv::Mat;
using FPRUS::EigensketchSST;

void eigensketchsst_test()
{
	const string &trainingPhotoPath =
		"../DataSet/CUHK_training_photo/all_filename.txt";
	const string &trainingSketchPath =
		"../DataSet/CUHK_training_sketch/all_filename.txt";
	const string &trainingFPPPath =
		"../DataSet/CUHK_training_faducial_points_photo/all_filename.txt";
	const string &trainingFPSPath =
		"../DataSet/CUHK_training_faducial_points_sketch/all_filename.txt";

	EigensketchSST esst(88, 35);
	esst.loadTrainingImgs(trainingPhotoPath, trainingSketchPath);
	esst.loadTrainingShape(trainingFPPPath, trainingFPSPath);

	esst.computeEM();

	//esst.releaseTrainingShape();
	//esst.releaseTrainingImgs();
}