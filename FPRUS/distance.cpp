/*
* Author: Corfox
* Date: 2015.10.22
*/

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\nonfree\nonfree.hpp>
#include <iostream>
#include <string>
#include <fstream>
#include <cstdlib>
#include <assert.h>
#include <vector>
#include "recognition.h"


using std::string;
using std::ifstream;
using std::ofstream;
using std::vector;
using cv::Mat;
using cv::imread;

namespace FPRUS {

	Distance::Distance()
	{
		this->trainImgNumbers = 0;
		this->trainingSet = new TrainItem();
	}

	Distance::~Distance()
	{
		while (this->trainingSet != NULL)
		{
			this->trainingSet->trainImg.release();
			this->trainingSet->feature.release();
			TrainItem *tmp = this->trainingSet;
			this->trainingSet = this->trainingSet->next;
			delete tmp;
		}
	}

	/* 载入训练集
	* @para fileName fileName文件中保存了所有的训练图像的名字
	*/
	void Distance::loadTrainingImgs(const string &fileName)
	{
		ifstream fin(fileName, ifstream::in);
		string imgName;
		TrainItem *currentNode = this->trainingSet;
		
		while (!fin.eof())
		{
			fin >> imgName;
			if (imgName.empty())
				continue;
			currentNode->next = new TrainItem(); //创建链表结点
			currentNode->next->prev = currentNode; 
			currentNode = currentNode->next;
			currentNode->trainImg = imread(imgName, CV_LOAD_IMAGE_GRAYSCALE);
			currentNode->trainImg.convertTo(currentNode->trainImg, CV_64FC1);
			currentNode->itemLabel = 
				imgName.substr(imgName.find_last_of('/') + 1, imgName.length());
			this->trainImgNumbers++; //加载的训练集图像数目

			// 断言训练集中的每幅图像都有相同的大小
			assert(this->trainingSet->next->trainImg.rows == currentNode->trainImg.rows
				&& this->trainingSet->next->trainImg.cols == currentNode->trainImg.cols);
			imgName.clear();
		}

		fin.close();
	} // end loadTrainingImgs(const string&)

	/* 计算距离，计算后的距离按从小到大排列
	* @para queryImg 查询图像
	* @para distanceType 计算距离类型，EUCLID_DISTANCE
	*/
	void Distance::computeDistance(const Mat &query, int distanceType)
	{
		// 断言已经加载训练集图像，并且查询图像的大小与训练集中的图像大小相同
		assert(this->trainingSet->next != NULL
			&& query.rows == this->trainingSet->next->trainImg.rows
			&& query.cols == this->trainingSet->next->trainImg.cols);

		Mat queryImg = query;
		queryImg.convertTo(queryImg, CV_64FC1);
		TrainItem *currentNode = this->trainingSet;

		switch (distanceType)
		{
		case Distance::EUCLID_DISTANCE:
			while (currentNode->next != NULL)
			{
				currentNode = currentNode->next;
				currentNode->distance = cv::norm(currentNode->trainImg, queryImg,
					cv::NORM_L2);
			}
			break;
		case Distance::SIFT_DISTANCE:
			this->computeSiftDistance(this->trainingSet, queryImg);
			break;
		default:
			break;
		}

		currentNode = this->trainingSet;
		TrainItem tmpTrainItem;
		while (currentNode->next != NULL) //选择排序
		{
			currentNode = currentNode->next;
			TrainItem *minNode = currentNode;
			TrainItem *tmpNode = currentNode->next;

			while (tmpNode != NULL) //找到后续节点的最小值
			{
				if (tmpNode->distance < minNode->distance)
					minNode = tmpNode;
				tmpNode = tmpNode->next;
			}

			//交换最小值节点与当前节点的内容，即trainImg, distance, itemLabel
			//但不交换链表的前后指正
			tmpTrainItem.trainImg = minNode->trainImg;
			tmpTrainItem.feature = minNode->feature;
			tmpTrainItem.distance = minNode->distance;
			tmpTrainItem.itemLabel = minNode->itemLabel;
			minNode->trainImg = currentNode->trainImg;
			minNode->feature = currentNode->feature;
			minNode->distance = currentNode->distance;
			minNode->itemLabel = currentNode->itemLabel;
			currentNode->trainImg = tmpTrainItem.trainImg;
			currentNode->feature = tmpTrainItem.feature;
			currentNode->distance = tmpTrainItem.distance;
			currentNode->itemLabel = tmpTrainItem.itemLabel;
		}

	} // end computeDistance(const Mat&, int)

	/* 获得第n小的距离
	* @para n 第n小的距离，下标从1开始
	* @return 返回第n小的距离
	*/
	double Distance::getDistance(int n) const
	{
		assert(n > 0 && n <= this->trainImgNumbers);

		TrainItem *tmpNode = this->trainingSet;
		for (int i = n; i > 0; --i)
			tmpNode = tmpNode->next;

		return tmpNode->distance;
	}

	/* 获得距离第n小的图像的标号，标号即加载图像时的图像名
	* @para n 第n小的图像的标号，下标从1开始
	* @return string 返回第n小的图像的标号
	*/
	string Distance::getLabel(int n) const
	{
		assert(n > 0 && n <= this->trainImgNumbers);

		TrainItem *tmpNode = this->trainingSet;
		for (int i = n; i > 0; --i)
			tmpNode = tmpNode->next;

		return tmpNode->itemLabel;
	}

	/* 获取第n张训练集图像
	* @para n 第n张训练集图像，下标从1开始
	* @return Mat 训练集图像
	*/
	Mat Distance::getTrainingImg(int n) const
	{
		assert(n > 0 && n <= this->trainImgNumbers);

		TrainItem *tmpNode = this->trainingSet;
		for (int i = n; i > 0; --i)
			tmpNode = tmpNode->next;

		Mat result;
		tmpNode->trainImg.convertTo(result, CV_8UC1);
		return result;
	}

	/* 获取加载的训练集图像数目
	* @return int 返回训练集中图像的数目
	*/
	int Distance::getTrainingNum() const
	{
		return this->trainImgNumbers;
	}


	/* 计算sift特征点之间的距离
	* @para trainSet 指向双向训练集链表的指针
	* @para queryImg 查询图像
	*/
	void Distance::computeSiftDistance(TrainItem *currentNode, const Mat &query)
	{
		if (this->trainingSet->next->feature.empty())
			this->extractSiftFeature(100);

		cv::SiftFeatureDetector detector(100);
		vector<cv::KeyPoint> queryKeypoints;
		cv::SiftDescriptorExtractor descriptor;
		Mat queryDescriptors;
		cv::BFMatcher matcher;
		vector<cv::DMatch> matches;

		Mat queryImg;
		query.convertTo(queryImg, CV_8UC1);
		detector.detect(queryImg, queryKeypoints);
		descriptor.compute(queryImg, queryKeypoints, queryDescriptors);
		queryDescriptors.convertTo(queryDescriptors, CV_32FC1);

		while (currentNode->next != NULL)
		{
			currentNode = currentNode->next;
			matcher.match(queryDescriptors, currentNode->feature, matches);

			if (matches.size() > 0)
				currentNode->distance = 0;
			else
				continue;

			for (int i = 0; i < matches.size(); ++i)
			{
				currentNode->distance += matches.at(i).distance;
			}
			currentNode->distance /= this->trainImgNumbers;

			matches.clear();
		}
	}

	/* 提取训练集每张图片的SIFT特征。
	 * @para number 每张图像提取的特征个数。
	 */
	void Distance::extractSiftFeature(int number)
	{
		TrainItem *currentNode = this->trainingSet;

		cv::SiftFeatureDetector detector(number);
		vector<cv::KeyPoint> trainKeypoints;
		cv::SiftDescriptorExtractor descriptor;

		Mat trainTmpImg;
		while (currentNode->next != NULL)
		{
			currentNode = currentNode->next;
			currentNode->trainImg.convertTo(trainTmpImg, CV_8UC1);
			detector.detect(trainTmpImg, trainKeypoints);
			descriptor.compute(trainTmpImg, trainKeypoints, 
				currentNode->feature);
			currentNode->feature.convertTo(currentNode->feature, CV_32FC1);
		}

	}

} //end namespace FPRUS