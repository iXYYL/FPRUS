/*
 * Author: Corfox
 * Date: 2015.10.22
 */

#include <string>
#include <fstream>
#include <cmath>
#include <stdio.h>
#include <assert.h>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include "transformation.h"

using std::string;
using std::ifstream;
using cv::Mat;

namespace FPRUS {
	
	Eigensketch::Eigensketch(int imgNumber)
	{
		this->IMG_NUM = imgNumber;
		trainingPhoto = NULL;
		trainingSketch = NULL;
		trainingPhotoVector = NULL;
		trainingSketchVector = NULL;
		cols = rows = 0;
	}

	Eigensketch::~Eigensketch()
	{
		if (trainingPhoto != NULL)
			delete[] trainingPhoto;
		if (trainingSketch != NULL)
			delete[] trainingSketch;
		if (trainingPhotoVector != NULL)
			delete trainingPhotoVector;
		if (trainingSketchVector != NULL)
			delete trainingSketchVector;
		photoMean.release();
		sketchMean.release();
		eigenValue.release();
		eigenVector.release();
		orthonormalMat.release();
		rows = cols = 0;
		trainingPhoto = trainingSketch = trainingPhotoVector = trainingSketchVector
			= NULL;
	}

	/* 加载训练相片集
	 * @para photoFilename 该文件中包含了训练集中的所有的图像的名字
	 */
	void Eigensketch::loadTrainingPhoto(const string &photoFilename)
	{
		assert(IMG_NUM > 0);
		if (trainingPhoto != NULL)
		{
			delete[] trainingPhoto;
			trainingPhoto = NULL;
		}

		trainingPhoto = new Mat[IMG_NUM];
				
		ifstream fin(photoFilename, ifstream::in);
		int i = 0;
		string imgName;
		while (!fin.eof())
		{
			fin >> imgName;
			if (imgName.empty())
				continue;
			*(trainingPhoto + i) = cv::imread(imgName, CV_LOAD_IMAGE_GRAYSCALE);
			//确保加载的训练集中的每一张图像具有相同的分辨率
			assert((*(trainingPhoto)).rows == (*(trainingPhoto + i)).rows
				&& (*(trainingPhoto)).cols == (*(trainingPhoto + i)).cols);
			(*(trainingPhoto + i)).convertTo((*(trainingPhoto + i)), CV_64FC1);
			imgName.clear();
			++i;
		}
		fin.close();

		this->rows = (*(trainingPhoto)).rows;
 		this->cols = (*(trainingPhoto)).cols;

	} //end loadTrainingPhoto

	/* 加载训练素描集
	 * @para sketchFileName 该文件中包含了训练集中所有的图像的名字*/
	void Eigensketch::loadTrainingSketch(const string &sketchFilename)
	{
		assert(IMG_NUM > 0);
		if (trainingSketch != NULL)
		{
			delete[] trainingSketch;
			trainingSketch = NULL;
		}

		trainingSketch = new Mat[IMG_NUM];

		int i = 0;
		string imgName;
		ifstream fin(sketchFilename, ifstream::in);

		while (!fin.eof())
		{
			fin >> imgName;
			if (imgName.empty())
				continue;
			*(trainingSketch + i) = cv::imread(imgName, CV_LOAD_IMAGE_GRAYSCALE);
			// 确保训练集中的每张图像都有相同的分辨率
			assert((*(trainingSketch)).cols == (*(trainingSketch + i)).cols
				&& (*(trainingSketch)).rows == (*(trainingSketch + i)).rows);
			(*(trainingSketch + i)).convertTo((*(trainingSketch + i)), CV_64FC1);
			imgName.clear();
			++i;
		}
		fin.close();
	} // end loadTrainingSketch

	/* 加载训练相片集与训练素描集
	 * @para photoFilename 该文件中包含了训练相片集中所有的图像的名字
	 * @para sketchFilename 该文件中包含了训练素描集中所有的图像的名字
	 */
	void Eigensketch::loadTrainingImg(const string &photoFilename,
		const string &sketchFilename)
	{
		assert(IMG_NUM > 0);

		// 加载训练相片集
		loadTrainingPhoto(photoFilename);
		//加载训练素描集
		loadTrainingSketch(sketchFilename);
	}

	/* 释放加载的训练相片集*/
	void Eigensketch::releaseTrainingPhoto()
	{
		if (trainingPhoto != NULL)
			delete[] trainingPhoto;

		trainingPhoto = NULL;
	}

	/* 释放加载的训练素描集*/
	void Eigensketch::releaseTrainingSketch()
	{
		if (trainingSketch != NULL)
			delete[] trainingSketch;

		trainingSketch = NULL;
	}

	/* 释放加载的训练相片集与素描集*/
	void Eigensketch::releaseTrainingImg()
	{
		if (trainingPhoto != NULL)
		{
			delete[] trainingPhoto;
			trainingPhoto = NULL;
		}
		if (trainingSketch != NULL)
		{
			delete[] trainingSketch;
			trainingSketch = NULL;
		}
	}

	/* 计算重构相片与重构素描的相关系数*/
	void Eigensketch::computeParameters()
	{
		if (!this->photoMean.empty()) //已经计算过均值
			return;

		//计算训练相片集的均值
		computeMean(FLAG_PHOTO);
		//计算训练素描集的均值
		computeMean(FLAG_SKETCH);

		//计算训练相片集的每张图片与均值的差值
		computeDifference(FLAG_PHOTO);
		//计算训练素描集的每张图片与均值的差值
		computeDifference(FLAG_SKETCH);

		//计算训练集的特征值与特征向量
		computeEigen();

		//计算训练集协方差矩阵的标准特征矩阵
		assert((*trainingPhotoVector).cols == eigenVector.rows
			&& eigenVector.cols == eigenValue.rows);
		Mat eigenValueSqrt = Mat::zeros(eigenValue.rows, eigenValue.rows, CV_64FC1);
		for (int m = 0; m < eigenValueSqrt.rows; ++m)
		for (int n = 0; n < eigenValueSqrt.cols; n++)
		{
			if (m == n && eigenValue.at<double>(m, 0) > 0)
			{
				eigenValueSqrt.at<double>(m, n)
					= 1.0 / std::sqrt(eigenValue.at<double>(m, 0));
			}
		}
		orthonormalMat = (*trainingPhotoVector) * eigenVector * eigenValueSqrt;

	}

	/* 重构相片
	* @para photo 待重构的相片
	* @return Mat 重构的相片*/
	Mat Eigensketch::reconstructPhoto(Mat &photo)
	{

		assert(!photo.empty());

		photo.convertTo(photo, CV_64FC1);

		Mat photoVector(photo.rows * photo.cols, 1, CV_64FC1);
		matrixToColVector(photoVector, photo);
		Mat photoMeanVector(photo.rows * photo.cols, 1, CV_64FC1);
		matrixToColVector(photoMeanVector, this->photoMean);

		//计算投影系数
		Mat projectionCoefficient = orthonormalMat.t() * (photoVector - photoMeanVector);

		//计算重构系数
		Mat eigenValueSqrt = Mat::zeros(eigenValue.rows, eigenValue.rows, CV_64FC1);
		for (int m = 0; m < eigenValueSqrt.rows; ++m)
		for (int n = 0; n < eigenValueSqrt.cols; ++n)
		{
			if (m == n && eigenValue.at<double>(m, 0) > 0)
			{
				eigenValueSqrt.at<double>(m, n)
					= 1.0 / std::sqrt(eigenValue.at<double>(m, 0));
			}
		}
		Mat reconstructionCoefficient = eigenVector * eigenValueSqrt * projectionCoefficient;

		//归一化重构系数
		double sum_coeff = 0;
		for (int i = 0; i < this->IMG_NUM; ++i)
		{
			sum_coeff += reconstructionCoefficient.at<double>(i, 0);
		}
		//reconstructionCoefficient = reconstructionCoefficient * (1.0 / sum_coeff);
		
		//重构图像
		Mat reconstructedPhoto(photo.rows, photo.cols, CV_64FC1);
		for (int i = 0; i < this->IMG_NUM; ++i)
		{
			reconstructedPhoto += (*(trainingPhoto + i)) *
				reconstructionCoefficient.at<double>(i, 0);
		}

		reconstructedPhoto = reconstructedPhoto + photoMean * (1 - sum_coeff);
		photo.convertTo(photo, CV_8UC1);
		reconstructedPhoto.convertTo(reconstructedPhoto, CV_8UC1);
		return reconstructedPhoto;
	}

	/* 重构素描
	* @para photo 待重构的相片
	* @return Mat 重构的素描图*/
	Mat Eigensketch::reconstructSketch(Mat &photo)
	{
		assert(!photo.empty());

		photo.convertTo(photo, CV_64FC1);

		if (!trainingSketch)
			printf("Please load the training sketch set!");

		Mat photoVector(photo.cols * photo.rows, 1, CV_64FC1);
		matrixToColVector(photoVector, photo);
		Mat photoMeanVector(photo.cols * photo.rows, 1, CV_64FC1);
		matrixToColVector(photoMeanVector, this->photoMean);
		
		//计算投影系数
		Mat projectionCoefficient = orthonormalMat.t() * (photoVector - photoMeanVector);

		//计算重构系数
		//Mat eigenValueSqrt = Mat::diag(eigenValue) * 0.5;
		Mat eigenValueSqrt = Mat::zeros(eigenValue.rows, eigenValue.rows, CV_64FC1);
		for (int m = 0; m < eigenValueSqrt.rows; ++m)
		{
			for (int n = 0; n < eigenValueSqrt.cols; ++n)
			{
				if (m == n && eigenValue.at<double>(m, 0) > 0)
				{
						eigenValueSqrt.at<double>(m, n) =
							1.0 / std::sqrt(eigenValue.at<double>(m, 0));
				}
			}
		}

		Mat reconstructionCoefficient = eigenVector * eigenValueSqrt * projectionCoefficient;

		//重构系数计算，测试
		Mat test = eigenVector * eigenValueSqrt * (eigenVector * eigenValueSqrt).t()
			* (*(trainingPhotoVector)).t() * (photoVector - photoMeanVector);

		//归一化重构系数
		double sum_coeff = 0;
		for (int i = 0; i < this->IMG_NUM; ++i)
		{
			sum_coeff += reconstructionCoefficient.at<double>(i, 0);
		}
		//reconstructionCoefficient = reconstructionCoefficient * (1.0 / sum_coeff);


		//重构图像
		Mat reconstructedSketch = Mat::zeros((*trainingSketch).rows,
			(*trainingSketch).cols, CV_64FC1);
		for (int i = 0; i < this->IMG_NUM; ++i)
		{
			reconstructedSketch += (*(trainingSketch + i)) *
				reconstructionCoefficient.at<double>(i, 0);
		}

		reconstructedSketch = reconstructedSketch + sketchMean * (1 - sum_coeff);

		photo.convertTo(photo, CV_8UC1);
		//cv::normalize(reconstructedSketch, reconstructedSketch, 0, 256, cv::NORM_MINMAX);
		reconstructedSketch.convertTo(reconstructedSketch, CV_8UC1);
		return reconstructedSketch;
	}

	/* 将矩阵转换列矢量，矩阵按行展开*/
	void Eigensketch::matrixToColVector(Mat &vector, const Mat &matrix)
	{
		assert(!vector.empty()
			&& vector.rows == (matrix.cols * matrix.rows));

		for (int i = 0; i < matrix.rows; ++i)
		{
			for (int j = 0; j < matrix.cols; ++j)
			{
				vector.at<double>(i * matrix.cols + j, 0) = matrix.at<double>(i, j);
			}
		}
	}

	/* 将列矢量转换为矩阵，矩阵的行与列是训练相片集中相片的列和宽*/
	void Eigensketch::colVectorToMatrix(Mat &matrix, const Mat &vector)
	{
		assert(!matrix.empty()
			&& vector.rows == (matrix.cols * matrix.rows));

		for (int i = 0; i < matrix.rows; ++i)
		{
			for (int j = 0; j < matrix.cols; ++j)
			{
				matrix.at<double>(i, j) = vector.at<double>(i * matrix.cols + j, 0);
			}
		}
	}

	/* 计算训练集的均值
   	 * @para flag 计算训练相片集或素描集的标志，FLAG_PHOTO FLAG_SKETCH FLAG_PHOTO_SKETCH
	 */
	void Eigensketch::computeMean(int flag)
	{
		switch (flag)
		{
		case FLAG_PHOTO:
			photoMean = Mat::zeros(this->rows, this->cols, CV_64FC1);
			for (int k = 0; k < this->IMG_NUM; ++k)
			{
				photoMean += *(trainingPhoto + k);
			}
			photoMean = photoMean * (1.0 / this->IMG_NUM);
			break;
		case FLAG_SKETCH:
			sketchMean = Mat::zeros((*trainingSketch).rows, (*trainingSketch).cols
				, CV_64FC1);
			for (int k = 0; k < this->IMG_NUM; ++k)
			{
				sketchMean += *(trainingSketch + k);
			}
			sketchMean = sketchMean * (1.0 / this->IMG_NUM);
			break;
		case FLAG_PHOTO_SKETCH:
			photoMean = Mat::zeros(this->rows, this->cols, CV_64FC1);
			sketchMean = Mat::zeros((*trainingSketch).rows, (*trainingSketch).cols
				, CV_64FC1);
			for (int k = 0; k < this->IMG_NUM; ++k)
			{
				photoMean += (*trainingPhoto + k);
				sketchMean += *(trainingSketch + k);
			}
			photoMean = photoMean * (1.0 / this->IMG_NUM);
			sketchMean = sketchMean * (1.0 / this->IMG_NUM);
			break;
		default:
			break;
		}
	} //end computeMean(Mat&, const Mat&)

	/* 计算训练集每张图像与均值的差值*/
	void Eigensketch::computeDifference(int flag)
	{
		// 将均值转换为列向量
		Mat photoMeanVector(this->cols * this->rows, 1, CV_64FC1);
		Mat sketchMeanVector(this->cols * this->rows, 1, CV_64FC1);

		switch (flag)
		{
		case FLAG_PHOTO:
			matrixToColVector(photoMeanVector, photoMean);
			trainingPhotoVector = new Mat(this->cols * this->rows, IMG_NUM, CV_64FC1);
			for (int k = 0; k < this->IMG_NUM; ++k)
			{
				matrixToColVector((*trainingPhotoVector).col(k), *(trainingPhoto + k));
				(*trainingPhotoVector).col(k) -= photoMeanVector;
			}
			break;
		case FLAG_SKETCH:
			matrixToColVector(sketchMeanVector, sketchMean);
			trainingSketchVector = new Mat(this->cols * this->rows, IMG_NUM, CV_64FC1);
			for (int k = 0; k < this->IMG_NUM; ++k)
			{
				matrixToColVector((*trainingSketchVector).col(k), *(trainingSketch + k));
				(*trainingSketchVector).col(k) -= sketchMeanVector;
			}
			break;
		case FLAG_PHOTO_SKETCH:
			matrixToColVector(photoMeanVector, photoMean);
			matrixToColVector(sketchMeanVector, sketchMean);
			trainingPhotoVector = new Mat(this->cols * this->rows, IMG_NUM, CV_64FC1);
			trainingSketchVector = new Mat(this->cols * this->rows, IMG_NUM, CV_64FC1);
			for (int k = 0; k < this->IMG_NUM; ++k)
			{
				matrixToColVector((*trainingPhotoVector).col(k), *(trainingPhoto + k));
				matrixToColVector((*trainingSketchVector).col(k), *(trainingSketch + k));
				(*trainingPhotoVector).col(k) -= photoMeanVector;
				(*trainingSketchVector).col(k) -= sketchMeanVector;
			}
			break;
		default:
			break;
		}
	} //end computeDifference(int)

	/* 计算协方差矩阵的特征值与特征向量*/
	void Eigensketch::computeEigen()
	{
		Mat covar = (*trainingPhotoVector).t() * (*trainingPhotoVector);
		cv::eigen(covar, eigenValue, eigenVector);

		//特征向量是按行存储的，特征值是一个列矢量，按由大到小的顺序排列
		//每个特征向量对应一个特征值，还是应该将其转置
		eigenVector = eigenVector.t();
		//if (eigenValue.at<double>(this->IMG_NUM - 1, 0) < 0)
		//	eigenValue.at<double>(this->IMG_NUM - 1, 0) = 0;
	}
} // end namespace FPRUS