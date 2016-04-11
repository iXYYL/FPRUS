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

	/* ����ѵ����
	* @para fileName fileName�ļ��б��������е�ѵ��ͼ�������
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
			currentNode->next = new TrainItem(); //����������
			currentNode->next->prev = currentNode; 
			currentNode = currentNode->next;
			currentNode->trainImg = imread(imgName, CV_LOAD_IMAGE_GRAYSCALE);
			currentNode->trainImg.convertTo(currentNode->trainImg, CV_64FC1);
			currentNode->itemLabel = 
				imgName.substr(imgName.find_last_of('/') + 1, imgName.length());
			this->trainImgNumbers++; //���ص�ѵ����ͼ����Ŀ

			// ����ѵ�����е�ÿ��ͼ������ͬ�Ĵ�С
			assert(this->trainingSet->next->trainImg.rows == currentNode->trainImg.rows
				&& this->trainingSet->next->trainImg.cols == currentNode->trainImg.cols);
			imgName.clear();
		}

		fin.close();
	} // end loadTrainingImgs(const string&)

	/* ������룬�����ľ��밴��С��������
	* @para queryImg ��ѯͼ��
	* @para distanceType ����������ͣ�EUCLID_DISTANCE
	*/
	void Distance::computeDistance(const Mat &query, int distanceType)
	{
		// �����Ѿ�����ѵ����ͼ�񣬲��Ҳ�ѯͼ��Ĵ�С��ѵ�����е�ͼ���С��ͬ
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
		while (currentNode->next != NULL) //ѡ������
		{
			currentNode = currentNode->next;
			TrainItem *minNode = currentNode;
			TrainItem *tmpNode = currentNode->next;

			while (tmpNode != NULL) //�ҵ������ڵ����Сֵ
			{
				if (tmpNode->distance < minNode->distance)
					minNode = tmpNode;
				tmpNode = tmpNode->next;
			}

			//������Сֵ�ڵ��뵱ǰ�ڵ�����ݣ���trainImg, distance, itemLabel
			//�������������ǰ��ָ��
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

	/* ��õ�nС�ľ���
	* @para n ��nС�ľ��룬�±��1��ʼ
	* @return ���ص�nС�ľ���
	*/
	double Distance::getDistance(int n) const
	{
		assert(n > 0 && n <= this->trainImgNumbers);

		TrainItem *tmpNode = this->trainingSet;
		for (int i = n; i > 0; --i)
			tmpNode = tmpNode->next;

		return tmpNode->distance;
	}

	/* ��þ����nС��ͼ��ı�ţ���ż�����ͼ��ʱ��ͼ����
	* @para n ��nС��ͼ��ı�ţ��±��1��ʼ
	* @return string ���ص�nС��ͼ��ı��
	*/
	string Distance::getLabel(int n) const
	{
		assert(n > 0 && n <= this->trainImgNumbers);

		TrainItem *tmpNode = this->trainingSet;
		for (int i = n; i > 0; --i)
			tmpNode = tmpNode->next;

		return tmpNode->itemLabel;
	}

	/* ��ȡ��n��ѵ����ͼ��
	* @para n ��n��ѵ����ͼ���±��1��ʼ
	* @return Mat ѵ����ͼ��
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

	/* ��ȡ���ص�ѵ����ͼ����Ŀ
	* @return int ����ѵ������ͼ�����Ŀ
	*/
	int Distance::getTrainingNum() const
	{
		return this->trainImgNumbers;
	}


	/* ����sift������֮��ľ���
	* @para trainSet ָ��˫��ѵ���������ָ��
	* @para queryImg ��ѯͼ��
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

	/* ��ȡѵ����ÿ��ͼƬ��SIFT������
	 * @para number ÿ��ͼ����ȡ������������
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