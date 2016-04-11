/* 
 * Author: Corfox
 * Date: 2015.11.09
 */

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <string>
#include <iostream>
#include <fstream>
#include "transformation.h"

using std::ifstream;
using std::string;
using std::cout;
using cv::Mat;
using FPRUS::DataSet;

void dataset_test()
{
	int IMG_NUM = 88;
	const string &photoFilename = "../DataSet/CUHK_training_photo/all_filename.txt";
	const string &sketchFilename = "../DataSet/CUHK_training_sketch/all_filename.txt";
	
	DataSet dataSet(IMG_NUM);
	dataSet.loadTrainingPhotos(photoFilename);
	dataSet.loadTrainingSketches(sketchFilename);
	//dataSet.loadTrainingImgs(photoFilename, sketchFilename);
	Mat photoMean = dataSet.computeMean(DataSet::FLAG_PHOTO);
	Mat sketchMean = dataSet.computeMean(DataSet::FLAG_SKETCH);

	Mat matPhotoVector;
	Mat matSketchVector;
	dataSet.matrixToColVector(matPhotoVector, DataSet::FLAG_PHOTO);
	dataSet.matrixToColVector(matSketchVector, DataSet::FLAG_SKETCH);

	cout << "相片矢量的维数，相片矢量矩阵的行列数："
		<< matPhotoVector.col(0).rows << ","
		<< matPhotoVector.rows << "x" << matPhotoVector.cols << endl;
	cout << "素描矢量的维数，素描矢量矩阵的行列数："
		<< matSketchVector.col(0).rows << ","
		<< matSketchVector.rows << "x" << matSketchVector.cols << endl;
	dataSet.releaseTrainingImgs();

	photoMean.convertTo(photoMean, CV_8UC1);
	sketchMean.convertTo(sketchMean, CV_8UC1);
	cv::imshow("Photo Mean", photoMean);
	cv::imshow("Sketch Mean", sketchMean);
	cv::waitKey(0);
}
