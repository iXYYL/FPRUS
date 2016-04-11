/* 
 * Author: Corfox
 * Date: 2015.11.09
 */

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <string>
#include <fstream>
#include <assert.h>
#include "transformation.h"

using std::ifstream;
using std::string;
using cv::Mat;

namespace FPRUS {

	/* 构造函数
	 * @para imgNum 训练集大小，即训练集图像数目
	 */
	DataSet::DataSet(int imgNum)
	{
		assert(imgNum > 0);
		this->trainingPhoto = NULL;
		this->trainingSketch = NULL;
		this->imgNum = imgNum;
	}

	DataSet::~DataSet()
	{
		if (this->trainingPhoto != NULL)
		{
			for (int i = 0; i < this->imgNum; ++i)
				(*(this->trainingPhoto + i)).release();
			delete[] this->trainingPhoto;
		}
		if (this->trainingSketch != NULL)
		{
			for (int i = 0; i < this->imgNum; ++i)
				(*(this->trainingSketch + i)).release();
			delete[] this->trainingSketch;
		}
		this->imgNum = 0;
	}

	/* 加载训练相片集
	* @para photoFileName 该文件中保存了训练相片集中的所有相片的相对路径与图像名
	*/
	void DataSet::loadTrainingPhotos(const string &photoFilename)
	{
		this->trainingPhoto = new Mat[this->imgNum];
		ifstream fin(photoFilename, ifstream::in);
		string imgName;

		int i = 0;
		while (!fin.eof())
		{
			fin >> imgName;
			if (imgName.empty())
				continue;
			
			*(this->trainingPhoto + i) = cv::imread(imgName, CV_LOAD_IMAGE_GRAYSCALE);
			(*(this->trainingPhoto + i)).convertTo(*(this->trainingPhoto + i), CV_64FC1);

			assert((*this->trainingPhoto).rows == (*(this->trainingPhoto + i)).rows
				&& (*this->trainingPhoto).cols == (*(this->trainingPhoto + i)).cols);

			++i;
			imgName.clear();
		}
		fin.close();
	}

	/* 释放训练相片集的内存
	*/
	void DataSet::releaseTrainingPhotos()
	{
		if (this->trainingPhoto != NULL)
		{
			for (int i = 0; i < this->imgNum; ++i)
			{
				(*(this->trainingPhoto + i)).release();
			}
			delete[] this->trainingPhoto;
			this->trainingPhoto = NULL;
		}
	}

	/* 加载训练素描集
	* @para sketchFilename 该文件中保存了训练素描集中的所有素描的相对路径与图像名
	*/
	void DataSet::loadTrainingSketches(const string &sketchFilename)
	{
		this->trainingSketch = new Mat[this->imgNum];
		ifstream fin(sketchFilename, ifstream::in);
		string imgName;

		int i = 0;
		while (!fin.eof())
		{
			fin >> imgName;
			if (imgName.empty())
				continue;

			*(this->trainingSketch + i) = cv::imread(imgName, CV_LOAD_IMAGE_GRAYSCALE);
			(*(this->trainingSketch + i)).convertTo(*(this->trainingSketch + i), CV_64FC1);

			assert((*this->trainingSketch).rows == (*(this->trainingSketch + i)).rows
				&& (*this->trainingSketch).cols == (*(this->trainingSketch + i)).cols);

			++i;
			imgName.clear();
		}
		fin.close();
	}

	/* 释放训练素描集的内存
	*/
	void DataSet::releaseTrainingSketches()
	{
		if (this->trainingSketch != NULL)
		{
			for (int i = 0; i < this->imgNum; ++i)
			{
				(*(this->trainingSketch + i)).release();
			}

			delete[] this->trainingSketch;
			this->trainingSketch = NULL;
		}
	}

	/* 加载训练集
	* @para photoFileName 该文件中保存了训练相片集中所有相片图的相对路径与图像名
	* @para sketchFilename 该文件中保存了训练素描集中的所有素描图的相对路径与图像名
	*/
	void DataSet::loadTrainingImgs(const string &photoFilename, const string &sketchFilname)
	{
		this->loadTrainingPhotos(photoFilename);
		this->loadTrainingSketches(sketchFilname);
	}

	/* 释放训练集的内存
	*/
	void DataSet::releaseTrainingImgs()
	{
		this->releaseTrainingPhotos();
		this->releaseTrainingSketches();
	}

	/* @return int 得到训练集中的图像像素行数
	*/
	int DataSet::getRows()
	{
		return (*this->trainingPhoto).rows;
	}

	/* @return int 得到训练集中图像的像素列数
	*/
	int DataSet::getCols()
	{
		return (*this->trainingPhoto).cols;
	}

	/* @return int 得到训练集中图像的像素数
	*/
	int DataSet::getPixelNums()
	{
		return (*this->trainingPhoto).rows * (*this->trainingPhoto).cols;
	}

	/* 计算训练集的均值
	* @para flag DataSet::FLAG_PHOTO计算相片集的均值
	*            DataSet::FLAG_SKETCH计算素描集的均值
	* @return Mat 返回相片集或素描集的均值
	*/
	Mat DataSet::computeMean(int flag)
	{
		Mat mean;
		if (DataSet::FLAG_PHOTO == flag)
		{
			mean = Mat::zeros((*this->trainingPhoto).rows,
				(*this->trainingPhoto).cols, CV_64FC1);
			for (int i = 0; i < this->imgNum; ++i)
				mean += *(this->trainingPhoto + i);
			mean = mean * (1.0 / this->imgNum);
		}
		else if (DataSet::FLAG_SKETCH == flag)
		{
			mean = Mat::zeros((*this->trainingSketch).rows,
				(*this->trainingSketch).cols, CV_64FC1);
			for (int i = 0; i < this->imgNum; ++i)
				mean += *(this->trainingSketch + i);
			mean = mean * (1.0 / this->imgNum);
		}
		
		return mean;
	}

	/* 将训练集的每张图片拉伸为列矢量
	* @para matVector 每张图片列矢量化后的结果
	* @para flag DataSet::FLAG_PHOTO将训练相片集每张图像拉伸为列矢量
	*            DataSet::FLAG_SKETCH将训练素描集每张图像拉伸为列矢量
	*/
	void DataSet::matrixToColVector(Mat &matVector, int flag)
	{

		switch (flag)
		{
		case DataSet::FLAG_PHOTO:
			matVector = Mat::zeros((*this->trainingPhoto).rows *
				(*this->trainingPhoto).cols, this->imgNum, CV_64FC1);

			for (int i = 0; i < this->imgNum; ++i)
			{
				for (int m = 0; m < (*this->trainingPhoto).rows; ++m)
				for (int n = 0; n < (*this->trainingPhoto).cols; ++n)
				{
					matVector.col(i).at<double>(m * (*this->trainingPhoto).cols + n, 0)
						= (*(this->trainingPhoto + i)).at<double>(m, n);
				}
			}
			break;
		case DataSet::FLAG_SKETCH:
			matVector = Mat::zeros((*this->trainingSketch).rows *
				(*this->trainingSketch).cols, this->imgNum, CV_64FC1);

			for (int i = 0; i < this->imgNum; ++i)
			{
				for (int m = 0; m < (*this->trainingSketch).rows; ++m)
				for (int n = 0; n < (*this->trainingSketch).cols; ++n)
				{
					matVector.col(i).at<double>(m * (*this->trainingSketch).cols + n, 0)
						= (*(this->trainingSketch + i)).at<double>(m, n);
				}
			}
			break;
		default:
			break;
		}
	}

	/* 计算指定数据集的均值
	* @para matSet 指向数据集的指针
	* @para mean 均值结果
	* @para matNum 数据集大小
	*/
	void DataSet::computeMean(const Mat *matSet, Mat &mean, int matNum)
	{
		assert(matSet != NULL);
		
		mean = Mat::zeros((*matSet).rows, (*matSet).cols, CV_64FC1);

		Mat tmp;
		for (int i = 0; i < matNum; ++i)
		{
			(*(matSet + i)).convertTo(tmp, CV_64FC1);
			mean += *(matSet + i);
		}

		mean = mean * (1.0 / matNum);
	}

	/* 将指定的矩阵拉伸为列矢量
	* @para matrix 待拉伸为列矢量的矩阵
	* @para matVector 矩阵列矢量化后的结果
	*/
	void DataSet::matrixToColVector(const Mat &matrix, Mat &matVector)
	{
		matVector = Mat::zeros(matrix.rows * matrix.cols, 1, CV_64FC1);

		Mat tmp;
		matrix.convertTo(tmp, CV_64FC1);
		for (int m = 0; m < matrix.rows; ++m)
		for (int n = 0; n < matrix.cols; ++n)
		{
			matVector.at<double>(m * matrix.cols + n, 0)
				= tmp.at<double>(m, n);
		}
	}

	/* 将指定的列矢量转换为矩阵，要求能从matrix得到行列信息
	* @para matVector 待转换为矩阵的列矢量
	* @para matrix 列矢量矩阵化的结果
	*/
	void DataSet::colVectorToMatrix(const Mat &matVector, Mat &matrix)
	{
		assert(matrix.rows > 0 && matrix.cols > 0 && 
			matVector.rows == matrix.rows * matrix.cols);

		for (int m = 0; m < matrix.rows; ++m)
		for (int n = 0; n < matrix.cols; ++n)
		{
			matrix.at<double>(m, n) = matVector.at<double>(m * matrix.cols + n, 0);
		}
	}

} // end namespace FPRUS