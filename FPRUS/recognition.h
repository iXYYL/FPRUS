/* 
 * Author: Corfox
 * Date: 2015.10.30
 */

#ifndef __REGOGNITION_H__
#define __RECOGNITION_H__

#include <opencv2\core\core.hpp>
#include <string>

using std::string;
using cv::Mat;

namespace FPRUS {

	/* 计算图像之间的距离，或图像特征之间的距离*/
	class Distance
	{
	public:
		const static int EUCLID_DISTANCE = 0;
		const static int SIFT_DISTANCE = 1;

	private:
		class TrainItem
		{
		public:
			Mat trainImg; //训练图像
			Mat feature; //特征
			double distance; //训练图像与查询图像之间的距离
			string itemLabel; //训练图像的标号，默认是图像名
			class TrainItem *next; //指向下一个训练图像的指针
			class TrainItem *prev; //指向上一个训练图像的指针

		public:
			TrainItem()
			{
				distance = 2.0E20;
				next = NULL;
				prev = NULL;
			}
			~TrainItem() {}
		};

	private: 
		//Mat queryImg; //查询图像
		int trainImgNumbers; //训练集中图像数目
		//指向训练图像集的指针，以双链表组织数据,该指针仅仅作为头结点不存储数据
		TrainItem *trainingSet;

	public:
		Distance();
		~Distance();

		/* 载入训练集
		 * @para fileName fileName文件中保存了所有的训练图像的名字  
		 */
		void loadTrainingImgs(const string &fileName);

		/* 计算距离，计算后的距离按从小到大排列
		 * @para queryImg 查询图像
		 * @para distanceType 计算距离类型，EUCLID_DISTANCE
		 */
		void computeDistance(const Mat &queryImg, int distanceType);

		/* 获得第n小的距离
		 * @para n 第n小的距离，下标从1开始
		 * @return double 返回第n小的距离
		 */
		double getDistance(int n) const;

		/* 获得距离第n小的图像的标号，标号即加载图像时的图像名
		 * @para n 第n小的图像的标号，下标从1开始
		 * @return string 返回第n小的图像的标号
		 */
		string getLabel(int n) const;

		/* 获取第n张训练集图像
		 * @para n 第n张训练集图像，下标从1开始
		 * @return Mat 训练集图像
		 */
		Mat getTrainingImg(int n) const;

		/* 获取加载的训练集图像数目
		 * @return int 返回训练集中图像的数目
		 */
		int getTrainingNum() const;

	private:
		/* 计算sift特征点之间的距离
		 * @para trainSet 指向双向训练集链表的指针
		 * @para queryImg 查询图像
		 */
		void computeSiftDistance(TrainItem *trainSet, const Mat &queryImg);

		/* 提取训练集每张图片的SIFT特征。
		 * @para number 每张图像提取的特征个数
		 */
		void extractSiftFeature(int number);
	}; // end class Distance
}

#endif // end __REGONITION_H__




