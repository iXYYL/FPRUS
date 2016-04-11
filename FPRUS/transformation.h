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

	/* 特征素描算法，将相片与素描特征变换到同一个特征空间。然后，
	 * 重构出输入相片或素描图的素描图
	 */
	class Eigensketch
	{
	private:
		Mat *trainingPhoto; // 训练相片集
		Mat *trainingSketch; // 训练素描集
		Mat *trainingPhotoVector; //训练相片集，每张图像展开成了列矢量
		Mat *trainingSketchVector; //训练素描集，每张图像展开成了列矢量
		Mat photoMean; // 相片集的均值
		Mat sketchMean; // 素描集的均值
		int IMG_NUM; // 训练集中图像的数目，素描集的图片数目应与相片集相同
		int cols; // 训练相片集中图像的像素列数，训练集中的每张图像应该有相同的分辨率
		int rows; // 训练相片集中图像的像素行数
		Mat orthonormalMat; //标准正交特征矩阵
		Mat eigenValue; //特征值
		Mat eigenVector; //特征矢量
		const static int FLAG_PHOTO = 0;
		const static int FLAG_SKETCH = 1;
		const static int FLAG_PHOTO_SKETCH = 2;

	public:
		Eigensketch(int imgNumber);
		~Eigensketch();

		/* 加载训练相片集
		 * @para photoFilename 该文件中包含了训练集中的所有的图像的名字
		 * 要求名字的对应顺序时一样的。
		 */
		void loadTrainingPhoto(const string &photoFilename);

		/* 加载训练素描集
		 * @para sketchFileName 该文件中包含了训练集中所有的图像的名字
		 * 要求名字的对应顺序时一样的。
		 */
		void loadTrainingSketch(const string &sketchFilename);

		/* 加载训练相片集与训练素描集
		 * @para photoFilename 该文件中包含了训练相片集中所有的图像的名字
		 * @para sketchFilename 该文件中包含了训练素描集中所有的图像的名字
		 * 要求名字的对应顺序时一样的。
		 */
		void loadTrainingImg(const string &photoFilename, const string &sketchFilename);

		/* 释放加载的训练相片集*/
		void releaseTrainingPhoto();

		/* 释放加载的训练素描集*/
		void releaseTrainingSketch();

		/* 释放加载的训练相片集与训练素描集*/
		void releaseTrainingImg();

		/* 计算用于重构相片与重构素描的相关系数*/
		void computeParameters();

		/* 重构相片
		 * @para photo 待重构的相片
		 * @return Mat 重构的相片*/
		Mat reconstructPhoto(Mat &photo);

		/* 重构素描
		 * @para photo 待重构的相片
		 * @return Mat 重构的素描图*/
		Mat reconstructSketch(Mat &photo);
	private:
		/* 将矩阵转换列矢量，矩阵按行展开*/
		void matrixToColVector(Mat &vector, const Mat &matrix);

		/* 将列矢量转换为矩阵，矩阵的行与列是训练相片集中相片的列和宽*/
		void colVectorToMatrix(Mat &matrix, const Mat &vector);

		/* 计算训练集的均值
		 * @para flag 计算训练相片集或素描集的标志，FLAG_PHOTO FLAG_SKETCH FLAG_PHOTO_SKETCH*/
		void computeMean(int flag);

		/* 计算训练集每张图像与与均值的差值*/
		void computeDifference(int flag);

		/* 计算协方差矩阵的特征值与特征矢量*/
		void computeEigen();

	}; // end class Eigensketch

	/* 之前设计类时考虑的不充分，没有抽象出算法的共有部分。
	 * 这里抽象出数据集，以及一些常用的对数据集的操作。
	 */
	class DataSet
	{
	protected:
		Mat *trainingPhoto; //训练相片集
		Mat *trainingSketch; //训练素描集
		int imgNum; //训练集中图像的数目，训练相片集与训练素描集中应该有相同的图像数目

	public:
		enum 
		{
			FLAG_PHOTO = 0,
			FLAG_SKETCH = 1
		};

	public:
		/* 构造函数
		 * @para imgNum 训练集大小，即训练集图像数目
		 */
		DataSet(int imgNum);
		virtual ~DataSet();

		/* 加载训练相片集
		 * @para photoFileName 该文件中保存了训练相片集中的所有相片的相对路径与图像名
		 * 要求名字的对应顺序时一样的。
		 */
		void loadTrainingPhotos(const string &photoFilename);

		/* 释放训练相片集的内存
		 */
		void releaseTrainingPhotos();
		
		/* 加载训练素描集
		 * @para sketchFilename 该文件中保存了训练素描集中的所有素描的相对路径与图像名
		 * 要求名字的对应顺序时一样的。
		 */
		void loadTrainingSketches(const string &sketchFilename);

		/* 释放训练素描集的内存
		 */
		void releaseTrainingSketches();

		/* 加载训练集
		 * @para photoFileName 该文件中保存了训练相片集中所有相片图的相对路径与图像名
		 * @para sketchFilename 该文件中保存了训练素描集中的所有素描图的相对路径与图像名
		 * 要求名字的对应顺序时一样的。
		 */
		void loadTrainingImgs(const string &photoFilename, const string &sketchFilname);

		/* 释放训练集的内存
		 */
		void releaseTrainingImgs();

		/* @return int 得到训练集中的图像像素行数
		 */
		int getRows();

		/* @return int 得到训练集中图像的像素列数
		 */
		int getCols();
		 
		/* @return int 得到训练集中图像的像素数
		 */
		int getPixelNums();
	
		/* 计算训练集的均值
		 * @para flag DataSet::FLAG_PHOTO计算相片集的均值
		 *            DataSet::FLAG_SKETCH计算素描集的均值
		 * @return Mat 返回相片集或素描集的均值
		 */
		virtual Mat computeMean(int flag);

		/* 将训练集的每张图片拉伸为列矢量
		 * @para matVector 每张图片列矢量化后的结果
		 * @para flag DataSet::FLAG_PHOTO将训练相片集每张图像拉伸为列矢量
		 *            DataSet::FLAG_SKETCH将训练素描集每张图像拉伸为列矢量
		 */
		virtual void matrixToColVector(Mat &matVector, int flag);

		/* 计算指定数据集的均值
		 * @para matSet 指向数据集的指针
		 * @para mean 均值结果
		 * @para matNum 数据集大小
		 */
		static void computeMean(const Mat *matSet, Mat &mean, int matNum);

		/* 将指定的矩阵拉伸为列矢量
		 * @para matrix 待拉伸为列矢量的矩阵
		 * @para matVector 矩阵列矢量化后的结果
		 */
		static void matrixToColVector(const Mat &matrix, Mat &matVector);

		/* 将指定的列矢量转换为矩阵，要求能从matrix得到行列信息
		 * @para matVector 待转换为矩阵的列矢量
		 * @para matrix 列矢量矩阵化的结果
		 */
		static void colVectorToMatrix(const Mat &matVector, Mat &matrix);
	}; //end class DataSet

	/* 特征素描算法，将相片与素描图像分离为形状与纹理两部分。然后，
	 * 分别进行特征变换，得到合成素描图。*/
	class EigensketchSST : public DataSet
	{
		vector<vector<cv::Point>> photoShape; //相片图的形状
		vector<vector<cv::Point>> sketchShape; //素描图的形状

		Mat *clippedPhoto; //裁剪的相片
		Mat *clippedSketch; //裁剪的素描
		Mat *photoTexture; //相片图的纹理
		Mat *sketchTexture; //素描图的纹理

		int pRows; //裁剪后的相片图的高度
		int pCols; //裁剪后的相片图的宽度
		int sRows; //裁剪后的素描图的高度
		int sCols; //裁剪后的素描图的宽度
		int fpNum; //每张图像的基准点数目

		vector<cv::Point> photoShapeMean; //相片图的形状均值
		Mat photoTextureMean; //相片图的纹理均值
		vector<cv::Point> sketchShapeMean; //素描图的形状均值
		Mat sketchTextureMean; //素描图的纹理均值
		Mat photoShapeEM; //相片集形状的特征矩阵
		Mat photoTextureEM; //素描集形状的特征矩阵
		Mat sketchShapeEM; //素描集形状的特征矩阵
		Mat sketchTextureEM; //素描集纹理的特征矩阵

	public:
		/* 构造函数
		 * @para imgNum 训练集大小，即训练集图像数目。
		 * @para fpNum 每张图像的基准点数目
		 */
		EigensketchSST(int imgNum, int fpNum);
		~EigensketchSST();

		/* 载入训练集的形状信息, 并计算相片形状均值与素描形状均值
		 * @para photoShapeFilename 该文件中保存了训练相片集的所有相片的形状信息的相对路径与文件名
		 * @para sketchShapeFilename 该文件中保存了训练素描集的所有素描的形状信息的相对路径与文件名
		 * 要求名字的对应顺序都是一样的。
		 */
		void loadTrainingShape(const string &photoShapeFilename, const string &sketchShapeFilename);

		/* 释放训练集形状信息的内存
		 */
		void releaseTrainingShape();

		/* 计算相片集纹理与形状，以及素描集纹理与形状的特征矩阵,
		 */
		void computeEM();

		/* 重构一幅图像
		 * @para img 
		 * @return Mat 重构结果
		 */
		Mat reconstructImg(const Mat &img);

	private:
		/* 计算裁剪的尺寸
		 * @para shapes 训练相片集或素描集的形状
		 * @para width 合适的裁剪宽度大小
		 * @para height 合适的裁剪高度大小
		 */
		void computeClippedSize(vector<vector<cv::Point>> &shapes, int &width, int &height);

		/* 裁剪图像
		 * @para src 输入单通道图像图像
		 * @para dst 裁剪图像，和src有相同的数据类型
		 * @para shape 裁剪的形状
		 * @para width 裁剪图像的宽度
		 * @para height 裁剪图像的高度
		 */
		void clipImg(const Mat &src, Mat &dst, const vector<cv::Point> &shape,
			const int &width, const int &height);

		/* 生成训练相片集与素描集的纹理*/
		void generateTexture();

	}; //end class EigensketchSST

} //end namespace FPRUS

#endif // end __TRANSFROMATION_H__