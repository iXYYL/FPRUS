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

	/* ���캯��
	 * @para imgNum ѵ������С����ѵ����ͼ����Ŀ
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

	/* ����ѵ����Ƭ��
	* @para photoFileName ���ļ��б�����ѵ����Ƭ���е�������Ƭ�����·����ͼ����
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

	/* �ͷ�ѵ����Ƭ�����ڴ�
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

	/* ����ѵ�����輯
	* @para sketchFilename ���ļ��б�����ѵ�����輯�е�������������·����ͼ����
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

	/* �ͷ�ѵ�����輯���ڴ�
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

	/* ����ѵ����
	* @para photoFileName ���ļ��б�����ѵ����Ƭ����������Ƭͼ�����·����ͼ����
	* @para sketchFilename ���ļ��б�����ѵ�����輯�е���������ͼ�����·����ͼ����
	*/
	void DataSet::loadTrainingImgs(const string &photoFilename, const string &sketchFilname)
	{
		this->loadTrainingPhotos(photoFilename);
		this->loadTrainingSketches(sketchFilname);
	}

	/* �ͷ�ѵ�������ڴ�
	*/
	void DataSet::releaseTrainingImgs()
	{
		this->releaseTrainingPhotos();
		this->releaseTrainingSketches();
	}

	/* @return int �õ�ѵ�����е�ͼ����������
	*/
	int DataSet::getRows()
	{
		return (*this->trainingPhoto).rows;
	}

	/* @return int �õ�ѵ������ͼ�����������
	*/
	int DataSet::getCols()
	{
		return (*this->trainingPhoto).cols;
	}

	/* @return int �õ�ѵ������ͼ���������
	*/
	int DataSet::getPixelNums()
	{
		return (*this->trainingPhoto).rows * (*this->trainingPhoto).cols;
	}

	/* ����ѵ�����ľ�ֵ
	* @para flag DataSet::FLAG_PHOTO������Ƭ���ľ�ֵ
	*            DataSet::FLAG_SKETCH�������輯�ľ�ֵ
	* @return Mat ������Ƭ�������輯�ľ�ֵ
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

	/* ��ѵ������ÿ��ͼƬ����Ϊ��ʸ��
	* @para matVector ÿ��ͼƬ��ʸ������Ľ��
	* @para flag DataSet::FLAG_PHOTO��ѵ����Ƭ��ÿ��ͼ������Ϊ��ʸ��
	*            DataSet::FLAG_SKETCH��ѵ�����輯ÿ��ͼ������Ϊ��ʸ��
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

	/* ����ָ�����ݼ��ľ�ֵ
	* @para matSet ָ�����ݼ���ָ��
	* @para mean ��ֵ���
	* @para matNum ���ݼ���С
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

	/* ��ָ���ľ�������Ϊ��ʸ��
	* @para matrix ������Ϊ��ʸ���ľ���
	* @para matVector ������ʸ������Ľ��
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

	/* ��ָ������ʸ��ת��Ϊ����Ҫ���ܴ�matrix�õ�������Ϣ
	* @para matVector ��ת��Ϊ�������ʸ��
	* @para matrix ��ʸ�����󻯵Ľ��
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