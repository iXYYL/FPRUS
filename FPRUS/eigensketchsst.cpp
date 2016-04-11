/* 
 * Author: Corfox
 * Date: 2015.11.09
 */

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\video\video.hpp>
#include <string>
#include <climits>
#include <fstream>
#include <assert.h>
#include <iostream>
#include "transformation.h"

using std::cout;
using std::endl;
using std::ifstream;
using std::string;
using cv::Mat;
using cv::imread;

namespace FPRUS {

	/* ���캯��
	* @para imgNum ѵ������С����ѵ����ͼ����Ŀ��
	* @para fpNum ÿ��ͼ��Ļ�׼����Ŀ
	*/
	EigensketchSST::EigensketchSST(int imgNum, int fpNum) : DataSet(imgNum)
	{
		this->photoTexture = this->sketchTexture = NULL;
		this->fpNum = fpNum;
	}
	
	EigensketchSST::~EigensketchSST()
	{

	}

	/* ����ѵ��������״��Ϣ, ��������Ƭ��״��ֵ��������״��ֵ
	* @para pshapeFilename ���ļ��б�����ѵ����Ƭ����������Ƭ����״��Ϣ�����·�����ļ���
	* @para sshapeFilename ���ļ��б�����ѵ�����輯�������������״��Ϣ�����·�����ļ���
	*/
	void EigensketchSST::loadTrainingShape(const string &pshapeFilename, const string &sshapeFilename)
	{
		assert(this->photoShape.empty() && this->sketchShape.empty());

		vector<cv::Point> tmpShape;
		vector<cv::Point> *tmpShapeMean = NULL;
		vector<vector<cv::Point>> *imgShape = NULL;
		ifstream fileFin;
		ifstream shapeFin;
		string shapeFilename;
		for (int k = 0; k < 2; ++k)
		{
			if (k == 0) //������Ƭ����״��Ϣ
			{
				imgShape = &this->photoShape;
				this->photoShapeMean.assign(this->fpNum, cv::Point(0, 0));
				tmpShapeMean = &this->photoShapeMean;
				fileFin.open(pshapeFilename, ifstream::in);
			}
			else //�����������״��Ϣ
			{
				imgShape = &this->sketchShape;
				this->sketchShapeMean.assign(this->fpNum, cv::Point(0, 0));
				tmpShapeMean = &this->sketchShapeMean;
				fileFin.open(sshapeFilename, ifstream::in);
			}

			while (!fileFin.eof())
			{
				fileFin >> shapeFilename;
				if (shapeFilename.empty())
					continue;

				shapeFin.open(shapeFilename, ifstream::in);
				float x, y;
				cv::Point tmpPoint;
				for (int i = 0; i < this->fpNum; ++i)
				{
					shapeFin >> x >> y;
					tmpPoint = cv::Point(x, y);
					tmpShape.push_back(tmpPoint);
					(*tmpShapeMean).at(i) += tmpPoint;
				}
				(*imgShape).push_back(tmpShape);
				tmpShape.clear();
				shapeFilename.clear();
				shapeFin.close();
			}
			
			imgShape = NULL;
			fileFin.close();
		}

		for (int i = 0; i < this->fpNum; ++i)
		{
			this->photoShapeMean.at(i) *= 1.0 / this->imgNum;
			this->sketchShapeMean.at(i) *= 1.0 / this->imgNum;
		}

	} //end loadTrainingShape(const string&, const string&)

	/* �ͷ�ѵ������״��Ϣ���ڴ�
	*/
	void EigensketchSST::releaseTrainingShape()
	{
		
	}

	/* ������Ƭ����������״���Լ����輯��������״����������,
	*/
	void EigensketchSST::computeEM()
	{
		this->generateTexture();
	}

	/* �ع�һ��ͼ��
	* @para img
	* @return Mat �ع����
	*/
	Mat EigensketchSST::reconstructImg(const Mat &img)
	{
		
		return Mat::zeros(2, 2, CV_64FC1);
	}

//////////////////////////////////////////////////////////////
//////					˽�г�Ա����				////////////////////////////
/////////////////////////////////////////////////////////////

	/* ����ü��ĳߴ�
	* @para shapes ѵ����Ƭ�������輯����״
	* @para width ���ʵĲü���ȴ�С
	* @para height ���ʵĲü��߶ȴ�С
	*/
	void EigensketchSST::computeClippedSize(vector<vector<cv::Point>> &shapes,
		int &width, int &height)
	{
		assert(!shapes.empty());

		int minX, maxX, minY, maxY;
		width = height = 0;

		for (int i = 0; i < shapes.size(); ++i)
		{
			assert(!shapes.at(i).empty());
			minX = minY = INT_MAX;
			maxX = maxY = INT_MIN;

			for (int j = 0; j < shapes.at(i).size(); ++j)
			{
				minX = shapes.at(i).at(j).x < minX ? shapes.at(i).at(j).x : minX;
				minY = shapes.at(i).at(j).y < minY ? shapes.at(i).at(j).y : minY;
				maxX = shapes.at(i).at(j).x > maxX ? shapes.at(i).at(j).x : maxX;
				maxY = shapes.at(i).at(j).y > maxY ? shapes.at(i).at(j).y : maxY;
			}

			width = (maxX - minX) > width ? maxX - minX : width;
			height = (maxY - minY) > height ? maxY - minY : height;
		}

	} //end computeClippedSize(vector<vector<cv::Point>>&, int&, int&)

	/* �ü�ͼ��
	* @para src ���뵥ͨ��ͼ��ͼ��, 
	* @para dst �ü�ͼ�񣬺�src����ͬ����������
	* @para width �ü�ͼ��Ŀ��
	* @para height �ü�ͼ��ĸ߶�
	*/
	void EigensketchSST::clipImg(const Mat &src, Mat &dst, const vector<cv::Point> &shape,
		const int &width, const int &height)
	{
		assert(!src.empty());

		vector<vector<cv::Point>> contours;
		vector<cv::Point> contour;
		cv::convexHull(shape, contour);
		contours.push_back(contour);

		Mat mask = Mat::zeros(src.rows, src.cols, CV_8UC1);
		cv::drawContours(mask, contours, 0, cv::Scalar(255), CV_FILLED);
		
		Mat tmpDst = Mat::zeros(src.rows, src.cols, CV_8UC1);
		src.copyTo(tmpDst, mask);

		int minX, minY, maxX, maxY;
		minX = minY = INT_MAX;
		maxX = maxY = INT_MIN;
		for (int i = 0; i < contour.size(); ++i)
		{
			minX = contour.at(i).x < minX ? contour.at(i).x : minX;
			minY = contour.at(i).y < minY ? contour.at(i).y : minY;
			maxX = contour.at(i).x > maxX ? contour.at(i).x : maxX;
			maxY = contour.at(i).y > maxY ? contour.at(i).y : maxY;
		}
		int midX = (maxX + minX) / 2;
		int midY = (maxY + minY) / 2;

		minY = midY - height / 2, maxY = midY + height / 2;
		minX = midX - width / 2, maxX = midX + width / 2;
		minY = minY < 0 ? 0 : minY, maxY = (maxY - minY) > height ? height : maxY;
		minX = minX < 0 ? 0 : minX, maxX = (maxX - minX) > width ? width : maxX;
		
		dst = tmpDst(cv::Range(minY, maxY), cv::Range(minX, maxX));
	}

	/* ����ѵ����Ƭ�������輯������*/
	void EigensketchSST::generateTexture()
	{
		//this->clippedPhoto = new Mat[this->imgNum];
		//this->clippedSketch = new Mat[this->imgNum];
		this->photoTexture = new Mat[this->imgNum];
		this->sketchTexture = new Mat[this->imgNum];
		Mat warpMat;
		cv::Size tmpSize;

		vector<vector<cv::Point>> shape;
		shape.push_back(this->photoShapeMean);
		this->computeClippedSize(shape, this->pCols, this->pRows);
		shape.clear();
		shape.push_back(this->sketchShapeMean);
		this->computeClippedSize(shape, this->sCols, this->sRows);

		tmpSize = cv::Size((*this->trainingPhoto).cols, (*this->trainingPhoto).rows);
		for (int i = 0; i < this->imgNum; ++i)
		{
			warpMat = cv::estimateRigidTransform(this->photoShape.at(i),
				this->photoShapeMean, true);
			cv::warpAffine(*(this->trainingPhoto + i), *(this->photoTexture + i),
				warpMat, tmpSize, cv::INTER_LINEAR);
			this->clipImg(*(this->photoTexture + i), *(this->photoTexture + i),
				this->photoShapeMean, this->pCols, this->pRows);
		}

		tmpSize = cv::Size((*this->trainingSketch).cols, (*this->trainingSketch).rows);
		for (int i = 0; i < this->imgNum; ++i)
		{
			warpMat = cv::estimateRigidTransform(this->sketchShape.at(i),
				this->sketchShapeMean, true);
			cv::warpAffine(*(this->trainingSketch + i), *(this->sketchTexture + i),
				warpMat, tmpSize, cv::INTER_LINEAR);
			this->clipImg(*(this->sketchTexture + i), *(this->sketchTexture + i),
				this->sketchShapeMean, this->sCols, this->sRows);
		}

		//cv::imwrite("clippedImg.jpg", *(this->clippedPhoto));
		//cv::imwrite("photoTexture.jpg", *(this->photoTexture));
		Mat tmp;
		for (int i = 0; i < this->imgNum; ++i)
		{
			(*(this->photoTexture + i)).convertTo(tmp, CV_8UC1);
			cv::imshow("Test", tmp);
			cv::waitKey(0);
		}
	} //end generateTexture()

} //end namespace FPRUS