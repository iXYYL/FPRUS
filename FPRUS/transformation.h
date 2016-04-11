/* 
 * Author: Corfox
 * Date: 2015.10.22
 */

#ifndef __TRANSFORMATION_H__
#define __TRANSFORMATION_H__

#include <string>
#include <opencv2\core\core.hpp>
#include <vector>

using std::string;
using std::vector;
using cv::Mat;

namespace FPRUS {

	/* ���������㷨������Ƭ�����������任��ͬһ�������ռ䡣Ȼ��
	 * �ع���������Ƭ������ͼ������ͼ
	 */
	class Eigensketch
	{
	private:
		Mat *trainingPhoto; // ѵ����Ƭ��
		Mat *trainingSketch; // ѵ�����輯
		Mat *trainingPhotoVector; //ѵ����Ƭ����ÿ��ͼ��չ��������ʸ��
		Mat *trainingSketchVector; //ѵ�����輯��ÿ��ͼ��չ��������ʸ��
		Mat photoMean; // ��Ƭ���ľ�ֵ
		Mat sketchMean; // ���輯�ľ�ֵ
		int IMG_NUM; // ѵ������ͼ�����Ŀ�����輯��ͼƬ��ĿӦ����Ƭ����ͬ
		int cols; // ѵ����Ƭ����ͼ�������������ѵ�����е�ÿ��ͼ��Ӧ������ͬ�ķֱ���
		int rows; // ѵ����Ƭ����ͼ�����������
		Mat orthonormalMat; //��׼������������
		Mat eigenValue; //����ֵ
		Mat eigenVector; //����ʸ��
		const static int FLAG_PHOTO = 0;
		const static int FLAG_SKETCH = 1;
		const static int FLAG_PHOTO_SKETCH = 2;

	public:
		Eigensketch(int imgNumber);
		~Eigensketch();

		/* ����ѵ����Ƭ��
		 * @para photoFilename ���ļ��а�����ѵ�����е����е�ͼ�������
		 * Ҫ�����ֵĶ�Ӧ˳��ʱһ���ġ�
		 */
		void loadTrainingPhoto(const string &photoFilename);

		/* ����ѵ�����輯
		 * @para sketchFileName ���ļ��а�����ѵ���������е�ͼ�������
		 * Ҫ�����ֵĶ�Ӧ˳��ʱһ���ġ�
		 */
		void loadTrainingSketch(const string &sketchFilename);

		/* ����ѵ����Ƭ����ѵ�����輯
		 * @para photoFilename ���ļ��а�����ѵ����Ƭ�������е�ͼ�������
		 * @para sketchFilename ���ļ��а�����ѵ�����輯�����е�ͼ�������
		 * Ҫ�����ֵĶ�Ӧ˳��ʱһ���ġ�
		 */
		void loadTrainingImg(const string &photoFilename, const string &sketchFilename);

		/* �ͷż��ص�ѵ����Ƭ��*/
		void releaseTrainingPhoto();

		/* �ͷż��ص�ѵ�����輯*/
		void releaseTrainingSketch();

		/* �ͷż��ص�ѵ����Ƭ����ѵ�����輯*/
		void releaseTrainingImg();

		/* ���������ع���Ƭ���ع���������ϵ��*/
		void computeParameters();

		/* �ع���Ƭ
		 * @para photo ���ع�����Ƭ
		 * @return Mat �ع�����Ƭ*/
		Mat reconstructPhoto(Mat &photo);

		/* �ع�����
		 * @para photo ���ع�����Ƭ
		 * @return Mat �ع�������ͼ*/
		Mat reconstructSketch(Mat &photo);
	private:
		/* ������ת����ʸ����������չ��*/
		void matrixToColVector(Mat &vector, const Mat &matrix);

		/* ����ʸ��ת��Ϊ���󣬾������������ѵ����Ƭ������Ƭ���кͿ�*/
		void colVectorToMatrix(Mat &matrix, const Mat &vector);

		/* ����ѵ�����ľ�ֵ
		 * @para flag ����ѵ����Ƭ�������輯�ı�־��FLAG_PHOTO FLAG_SKETCH FLAG_PHOTO_SKETCH*/
		void computeMean(int flag);

		/* ����ѵ����ÿ��ͼ�������ֵ�Ĳ�ֵ*/
		void computeDifference(int flag);

		/* ����Э������������ֵ������ʸ��*/
		void computeEigen();

	}; // end class Eigensketch

	/* ֮ǰ�����ʱ���ǵĲ���֣�û�г�����㷨�Ĺ��в��֡�
	 * �����������ݼ����Լ�һЩ���õĶ����ݼ��Ĳ�����
	 */
	class DataSet
	{
	protected:
		Mat *trainingPhoto; //ѵ����Ƭ��
		Mat *trainingSketch; //ѵ�����輯
		int imgNum; //ѵ������ͼ�����Ŀ��ѵ����Ƭ����ѵ�����輯��Ӧ������ͬ��ͼ����Ŀ

	public:
		enum 
		{
			FLAG_PHOTO = 0,
			FLAG_SKETCH = 1
		};

	public:
		/* ���캯��
		 * @para imgNum ѵ������С����ѵ����ͼ����Ŀ
		 */
		DataSet(int imgNum);
		virtual ~DataSet();

		/* ����ѵ����Ƭ��
		 * @para photoFileName ���ļ��б�����ѵ����Ƭ���е�������Ƭ�����·����ͼ����
		 * Ҫ�����ֵĶ�Ӧ˳��ʱһ���ġ�
		 */
		void loadTrainingPhotos(const string &photoFilename);

		/* �ͷ�ѵ����Ƭ�����ڴ�
		 */
		void releaseTrainingPhotos();
		
		/* ����ѵ�����輯
		 * @para sketchFilename ���ļ��б�����ѵ�����輯�е�������������·����ͼ����
		 * Ҫ�����ֵĶ�Ӧ˳��ʱһ���ġ�
		 */
		void loadTrainingSketches(const string &sketchFilename);

		/* �ͷ�ѵ�����輯���ڴ�
		 */
		void releaseTrainingSketches();

		/* ����ѵ����
		 * @para photoFileName ���ļ��б�����ѵ����Ƭ����������Ƭͼ�����·����ͼ����
		 * @para sketchFilename ���ļ��б�����ѵ�����輯�е���������ͼ�����·����ͼ����
		 * Ҫ�����ֵĶ�Ӧ˳��ʱһ���ġ�
		 */
		void loadTrainingImgs(const string &photoFilename, const string &sketchFilname);

		/* �ͷ�ѵ�������ڴ�
		 */
		void releaseTrainingImgs();

		/* @return int �õ�ѵ�����е�ͼ����������
		 */
		int getRows();

		/* @return int �õ�ѵ������ͼ�����������
		 */
		int getCols();
		 
		/* @return int �õ�ѵ������ͼ���������
		 */
		int getPixelNums();
	
		/* ����ѵ�����ľ�ֵ
		 * @para flag DataSet::FLAG_PHOTO������Ƭ���ľ�ֵ
		 *            DataSet::FLAG_SKETCH�������輯�ľ�ֵ
		 * @return Mat ������Ƭ�������輯�ľ�ֵ
		 */
		virtual Mat computeMean(int flag);

		/* ��ѵ������ÿ��ͼƬ����Ϊ��ʸ��
		 * @para matVector ÿ��ͼƬ��ʸ������Ľ��
		 * @para flag DataSet::FLAG_PHOTO��ѵ����Ƭ��ÿ��ͼ������Ϊ��ʸ��
		 *            DataSet::FLAG_SKETCH��ѵ�����輯ÿ��ͼ������Ϊ��ʸ��
		 */
		virtual void matrixToColVector(Mat &matVector, int flag);

		/* ����ָ�����ݼ��ľ�ֵ
		 * @para matSet ָ�����ݼ���ָ��
		 * @para mean ��ֵ���
		 * @para matNum ���ݼ���С
		 */
		static void computeMean(const Mat *matSet, Mat &mean, int matNum);

		/* ��ָ���ľ�������Ϊ��ʸ��
		 * @para matrix ������Ϊ��ʸ���ľ���
		 * @para matVector ������ʸ������Ľ��
		 */
		static void matrixToColVector(const Mat &matrix, Mat &matVector);

		/* ��ָ������ʸ��ת��Ϊ����Ҫ���ܴ�matrix�õ�������Ϣ
		 * @para matVector ��ת��Ϊ�������ʸ��
		 * @para matrix ��ʸ�����󻯵Ľ��
		 */
		static void colVectorToMatrix(const Mat &matVector, Mat &matrix);
	}; //end class DataSet

	/* ���������㷨������Ƭ������ͼ�����Ϊ��״�����������֡�Ȼ��
	 * �ֱ���������任���õ��ϳ�����ͼ��*/
	class EigensketchSST : public DataSet
	{
		vector<vector<cv::Point>> photoShape; //��Ƭͼ����״
		vector<vector<cv::Point>> sketchShape; //����ͼ����״

		Mat *clippedPhoto; //�ü�����Ƭ
		Mat *clippedSketch; //�ü�������
		Mat *photoTexture; //��Ƭͼ������
		Mat *sketchTexture; //����ͼ������

		int pRows; //�ü������Ƭͼ�ĸ߶�
		int pCols; //�ü������Ƭͼ�Ŀ��
		int sRows; //�ü��������ͼ�ĸ߶�
		int sCols; //�ü��������ͼ�Ŀ��
		int fpNum; //ÿ��ͼ��Ļ�׼����Ŀ

		vector<cv::Point> photoShapeMean; //��Ƭͼ����״��ֵ
		Mat photoTextureMean; //��Ƭͼ�������ֵ
		vector<cv::Point> sketchShapeMean; //����ͼ����״��ֵ
		Mat sketchTextureMean; //����ͼ�������ֵ
		Mat photoShapeEM; //��Ƭ����״����������
		Mat photoTextureEM; //���輯��״����������
		Mat sketchShapeEM; //���輯��״����������
		Mat sketchTextureEM; //���輯�������������

	public:
		/* ���캯��
		 * @para imgNum ѵ������С����ѵ����ͼ����Ŀ��
		 * @para fpNum ÿ��ͼ��Ļ�׼����Ŀ
		 */
		EigensketchSST(int imgNum, int fpNum);
		~EigensketchSST();

		/* ����ѵ��������״��Ϣ, ��������Ƭ��״��ֵ��������״��ֵ
		 * @para photoShapeFilename ���ļ��б�����ѵ����Ƭ����������Ƭ����״��Ϣ�����·�����ļ���
		 * @para sketchShapeFilename ���ļ��б�����ѵ�����輯�������������״��Ϣ�����·�����ļ���
		 * Ҫ�����ֵĶ�Ӧ˳����һ���ġ�
		 */
		void loadTrainingShape(const string &photoShapeFilename, const string &sketchShapeFilename);

		/* �ͷ�ѵ������״��Ϣ���ڴ�
		 */
		void releaseTrainingShape();

		/* ������Ƭ����������״���Լ����輯��������״����������,
		 */
		void computeEM();

		/* �ع�һ��ͼ��
		 * @para img 
		 * @return Mat �ع����
		 */
		Mat reconstructImg(const Mat &img);

	private:
		/* ����ü��ĳߴ�
		 * @para shapes ѵ����Ƭ�������輯����״
		 * @para width ���ʵĲü���ȴ�С
		 * @para height ���ʵĲü��߶ȴ�С
		 */
		void computeClippedSize(vector<vector<cv::Point>> &shapes, int &width, int &height);

		/* �ü�ͼ��
		 * @para src ���뵥ͨ��ͼ��ͼ��
		 * @para dst �ü�ͼ�񣬺�src����ͬ����������
		 * @para shape �ü�����״
		 * @para width �ü�ͼ��Ŀ��
		 * @para height �ü�ͼ��ĸ߶�
		 */
		void clipImg(const Mat &src, Mat &dst, const vector<cv::Point> &shape,
			const int &width, const int &height);

		/* ����ѵ����Ƭ�������輯������*/
		void generateTexture();

	}; //end class EigensketchSST

} //end namespace FPRUS

#endif // end __TRANSFROMATION_H__