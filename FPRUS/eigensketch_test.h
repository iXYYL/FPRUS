/* Author: Corfox
 * Date: 2015.10.30
 */

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <string>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <fstream>
#include <assert.h>
#include "transformation.h"
#include "recognition.h"

using cv::Mat;
using std::string;
using std::ifstream;
using std::ofstream;
using std::cin;
using std::cout;
using std::endl;
using std::vector;
using FPRUS::Eigensketch;

void eigensketch_test()
{
	//保存了训练相片集与素描集图像名的文件
	const string &training_cropped_photos = 
		"../imgName/CUHK_training_cropped_photos.txt";
	const string &training_cropped_sketches = 
		"../imgName/CUHK_training_cropped_sketches.txt";

	//训练集共有88张图片
	Eigensketch eigensketch(88);
	//加载训练相片集与素描集
	eigensketch.loadTrainingImg(training_cropped_photos, training_cropped_sketches);
	//计算相关的系数
	eigensketch.computeParameters();

	//保存了测试相片集与素描集图像名的文件
	const string &testing_cropped_photos =
		"../imgName/CUHK_testing_cropped_photos.txt";
	const string &testing_cropped_sketches =
		"../imgName/CUHK_testing_cropped_sketches.txt";
	//const string &testPhotoName = "../DataSet/CUHK_testing_cropped_photos/f1-001-01.jpg";
	//Mat testPhoto = cv::imread(testPhotoName, CV_LOAD_IMAGE_GRAYSCALE);
	//Mat reSketches = eigensketch.reconstructSketch(testPhoto);

	//保存测试相片(cropped)的重构素描图的文件夹名
	const string &reconstructed_testingps_dir =
		"../ReconstructedDataset/CUHK_testing_cropped_ps/";
	std::system("mkdir ..\\ReconstructedDataset\\CUHK_testing_cropped_ps\\");
	//保存测试素描(cropped)的重构素描图的文件夹名
	const string &reconstructed_testingss_dir =
		"../ReconstructedDataset/CUHK_testing_cropped_ss/";
	std::system("mkdir ..\\ReconstructedDataset\\CUHK_testing_cropped_ss\\");
	vector<int> compressionParams;
	compressionParams.push_back(CV_IMWRITE_JPEG_QUALITY);
	compressionParams.push_back(100);

	string testPhotoName;
	string testSketchName;
	Mat testPhoto;
	Mat testSketch;
	Mat reSketches;
	ifstream fin(testing_cropped_photos, ifstream::in);
	ifstream finSketches(testing_cropped_sketches, ifstream::in);
	while (!fin.eof())
	{
		fin >> testPhotoName;
		if (testPhotoName.empty())
			continue;

		finSketches >> testSketchName;
		if (testSketchName.empty())
			continue;

		testPhoto = cv::imread(testPhotoName, CV_LOAD_IMAGE_GRAYSCALE);
		testSketch = cv::imread(testSketchName, CV_LOAD_IMAGE_GRAYSCALE);

		reSketches = eigensketch.reconstructSketch(testPhoto);

		////写入重构的测试相片的素描图
		//string tmpImgName;
		//tmpImgName = testPhotoName.substr(testPhotoName.find_last_of("/") + 1,
		//	testPhotoName.find_last_of(".") - testPhotoName.find_last_of("/") - 1);
		//tmpImgName = reconstructed_testingps_dir + tmpImgName + "-sz1.jpg";
		//cout << "imwrite photo-to-sketch: " << tmpImgName << endl;
		//cv::imwrite(tmpImgName, reSketches, compressionParams);

		//reSketches = eigensketch.reconstructSketch(testSketch);
		////写入重构的测试素描的素描图
		//tmpImgName = testSketchName.substr(testSketchName.find_last_of("/") + 1,
		//	testSketchName.find_last_of(".") - testSketchName.find_last_of("/") - 1);
		//tmpImgName = reconstructed_testingss_dir + tmpImgName + ".jpg";
		//cout << "imwrite sketch-to-sketch: " << tmpImgName << endl;
		//cv::imwrite(tmpImgName, reSketches, compressionParams);

		testPhotoName.clear();
		testSketchName.clear();

		cv::imshow("testPhoto", testPhoto);
		cv::imshow("reSketches", reSketches);
		cv::waitKey(0);

		testPhoto.release();
		testSketch.release();
		reSketches.release();
	}

}

void eigensketch_recognition_test(int rankNumber)
{
	assert(rankNumber > 0);
	//测试集素描图的重构素描图，作为测试图像
	const string &ss_path =
		"../ReconstructedDataset/CUHK_testing_cropped_ss/all_filename.txt";
	//测试集相片图的重构素描图，作为训练集
	const string &ps_path =
		"../ReconstructedDataset/CUHK_testing_cropped_ps/all_filename.txt";

	FPRUS::Distance distance;
	distance.loadTrainingImgs(ps_path);
	cout << "load over!" << endl;

	ifstream fin(ss_path, ifstream::in);
	string queryImgName;
	Mat queryImg;
	int count = 0;
	while (!fin.eof())
	{
		fin >> queryImgName;
		if (queryImgName.empty())
			continue;

		queryImg = cv::imread(queryImgName, CV_LOAD_IMAGE_GRAYSCALE);
		distance.computeDistance(queryImg, FPRUS::Distance::EUCLID_DISTANCE);
		//double minDistance = distance.getDistance(1);
		queryImgName = queryImgName.substr(queryImgName.find_last_of('/') + 1,
			queryImgName.length());
		string minLabel;
		for (int i = 1; i < rankNumber + 1; ++i)
		{
			minLabel = distance.getLabel(i);
			if (minLabel.compare(queryImgName) == 0)
			{
				++count;
				break;
			}
		}
		queryImgName.clear();

		//cout << "最小距离：" << distance.getDistance(1) << endl;
		//cout << "最小距离的图像名：" << distance.getLabel(1) << endl;
		//Mat result = distance.getTrainingImg(1);
		//cv::imshow("result", result);
		//cv::waitKey(0);
	}

	cout << "基于Euclid Distance的识别率 (rank " << rankNumber << "):"
		<< (double)count / distance.getTrainingNum() << endl;

}

void eigensketch_sift_match_test(int rankNumber)
{
	assert(rankNumber > 0);

	//测试集原相片图
	const string &cropped_photos =
		"../DataSet/CUHK_testing_cropped_photos/all_filename.txt";
	//测试集原素描图
	const string &cropped_sketches =
		"../DataSet/CUHK_testing_cropped_sketches/all_filename.txt";

	FPRUS::Distance distance;
	distance.loadTrainingImgs(cropped_photos);
	cout << "load over!" << endl;


	ifstream fin(cropped_sketches, ifstream::in);
	string queryImgName;
	Mat queryImg;
	int count = 0;
	while (!fin.eof())
	{
		fin >> queryImgName;
		if (queryImgName.empty())
			continue;

		queryImg = cv::imread(queryImgName, CV_LOAD_IMAGE_GRAYSCALE);
		distance.computeDistance(queryImg, FPRUS::Distance::EUCLID_DISTANCE);
		//double minDistance = distance.getDistance(1);
		queryImgName = queryImgName.substr(queryImgName.find_last_of('/') + 1,
			queryImgName.find_last_of('-') - queryImgName.find_last_of('/') - 1);
		string minLabel;
		for (int i = 1; i < rankNumber + 1; ++i)
		{
			minLabel = distance.getLabel(i);
			minLabel = minLabel.substr(0, minLabel.find_last_of('.'));
			if (minLabel.compare(queryImgName) == 0)
			{
				++count;
				break;
			}
		}
		queryImgName.clear();

		//cout << "最小距离：" << distance.getDistance(1) << endl;
		//cout << "最小距离的图像名：" << distance.getLabel(1) << endl;
		//Mat result = distance.getTrainingImg(1);
		//cv::imshow("result", result);
		//cv::waitKey(0);
	}

	cout << "基于Euclid Distance的识别率 (rank " << rankNumber << "):"
		<< (double)count / distance.getTrainingNum() << endl;
}