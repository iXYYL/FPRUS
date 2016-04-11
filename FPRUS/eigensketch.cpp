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

	/* ����ѵ����Ƭ��
	 * @para photoFilename ���ļ��а�����ѵ�����е����е�ͼ�������
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
			//ȷ�����ص�ѵ�����е�ÿһ��ͼ�������ͬ�ķֱ���
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

	/* ����ѵ�����輯
	 * @para sketchFileName ���ļ��а�����ѵ���������е�ͼ�������*/
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
			// ȷ��ѵ�����е�ÿ��ͼ������ͬ�ķֱ���
			assert((*(trainingSketch)).cols == (*(trainingSketch + i)).cols
				&& (*(trainingSketch)).rows == (*(trainingSketch + i)).rows);
			(*(trainingSketch + i)).convertTo((*(trainingSketch + i)), CV_64FC1);
			imgName.clear();
			++i;
		}
		fin.close();
	} // end loadTrainingSketch

	/* ����ѵ����Ƭ����ѵ�����輯
	 * @para photoFilename ���ļ��а�����ѵ����Ƭ�������е�ͼ�������
	 * @para sketchFilename ���ļ��а�����ѵ�����輯�����е�ͼ�������
	 */
	void Eigensketch::loadTrainingImg(const string &photoFilename,
		const string &sketchFilename)
	{
		assert(IMG_NUM > 0);

		// ����ѵ����Ƭ��
		loadTrainingPhoto(photoFilename);
		//����ѵ�����輯
		loadTrainingSketch(sketchFilename);
	}

	/* �ͷż��ص�ѵ����Ƭ��*/
	void Eigensketch::releaseTrainingPhoto()
	{
		if (trainingPhoto != NULL)
			delete[] trainingPhoto;

		trainingPhoto = NULL;
	}

	/* �ͷż��ص�ѵ�����輯*/
	void Eigensketch::releaseTrainingSketch()
	{
		if (trainingSketch != NULL)
			delete[] trainingSketch;

		trainingSketch = NULL;
	}

	/* �ͷż��ص�ѵ����Ƭ�������輯*/
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

	/* �����ع���Ƭ���ع���������ϵ��*/
	void Eigensketch::computeParameters()
	{
		if (!this->photoMean.empty()) //�Ѿ��������ֵ
			return;

		//����ѵ����Ƭ���ľ�ֵ
		computeMean(FLAG_PHOTO);
		//����ѵ�����輯�ľ�ֵ
		computeMean(FLAG_SKETCH);

		//����ѵ����Ƭ����ÿ��ͼƬ���ֵ�Ĳ�ֵ
		computeDifference(FLAG_PHOTO);
		//����ѵ�����輯��ÿ��ͼƬ���ֵ�Ĳ�ֵ
		computeDifference(FLAG_SKETCH);

		//����ѵ����������ֵ����������
		computeEigen();

		//����ѵ����Э�������ı�׼��������
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

	/* �ع���Ƭ
	* @para photo ���ع�����Ƭ
	* @return Mat �ع�����Ƭ*/
	Mat Eigensketch::reconstructPhoto(Mat &photo)
	{

		assert(!photo.empty());

		photo.convertTo(photo, CV_64FC1);

		Mat photoVector(photo.rows * photo.cols, 1, CV_64FC1);
		matrixToColVector(photoVector, photo);
		Mat photoMeanVector(photo.rows * photo.cols, 1, CV_64FC1);
		matrixToColVector(photoMeanVector, this->photoMean);

		//����ͶӰϵ��
		Mat projectionCoefficient = orthonormalMat.t() * (photoVector - photoMeanVector);

		//�����ع�ϵ��
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

		//��һ���ع�ϵ��
		double sum_coeff = 0;
		for (int i = 0; i < this->IMG_NUM; ++i)
		{
			sum_coeff += reconstructionCoefficient.at<double>(i, 0);
		}
		//reconstructionCoefficient = reconstructionCoefficient * (1.0 / sum_coeff);
		
		//�ع�ͼ��
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

	/* �ع�����
	* @para photo ���ع�����Ƭ
	* @return Mat �ع�������ͼ*/
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
		
		//����ͶӰϵ��
		Mat projectionCoefficient = orthonormalMat.t() * (photoVector - photoMeanVector);

		//�����ع�ϵ��
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

		//�ع�ϵ�����㣬����
		Mat test = eigenVector * eigenValueSqrt * (eigenVector * eigenValueSqrt).t()
			* (*(trainingPhotoVector)).t() * (photoVector - photoMeanVector);

		//��һ���ع�ϵ��
		double sum_coeff = 0;
		for (int i = 0; i < this->IMG_NUM; ++i)
		{
			sum_coeff += reconstructionCoefficient.at<double>(i, 0);
		}
		//reconstructionCoefficient = reconstructionCoefficient * (1.0 / sum_coeff);


		//�ع�ͼ��
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

	/* ������ת����ʸ����������չ��*/
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

	/* ����ʸ��ת��Ϊ���󣬾������������ѵ����Ƭ������Ƭ���кͿ�*/
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

	/* ����ѵ�����ľ�ֵ
   	 * @para flag ����ѵ����Ƭ�������輯�ı�־��FLAG_PHOTO FLAG_SKETCH FLAG_PHOTO_SKETCH
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

	/* ����ѵ����ÿ��ͼ�����ֵ�Ĳ�ֵ*/
	void Eigensketch::computeDifference(int flag)
	{
		// ����ֵת��Ϊ������
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

	/* ����Э������������ֵ����������*/
	void Eigensketch::computeEigen()
	{
		Mat covar = (*trainingPhotoVector).t() * (*trainingPhotoVector);
		cv::eigen(covar, eigenValue, eigenVector);

		//���������ǰ��д洢�ģ�����ֵ��һ����ʸ�������ɴ�С��˳������
		//ÿ������������Ӧһ������ֵ������Ӧ�ý���ת��
		eigenVector = eigenVector.t();
		//if (eigenValue.at<double>(this->IMG_NUM - 1, 0) < 0)
		//	eigenValue.at<double>(this->IMG_NUM - 1, 0) = 0;
	}
} // end namespace FPRUS