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

	/* ����ͼ��֮��ľ��룬��ͼ������֮��ľ���*/
	class Distance
	{
	public:
		const static int EUCLID_DISTANCE = 0;
		const static int SIFT_DISTANCE = 1;

	private:
		class TrainItem
		{
		public:
			Mat trainImg; //ѵ��ͼ��
			Mat feature; //����
			double distance; //ѵ��ͼ�����ѯͼ��֮��ľ���
			string itemLabel; //ѵ��ͼ��ı�ţ�Ĭ����ͼ����
			class TrainItem *next; //ָ����һ��ѵ��ͼ���ָ��
			class TrainItem *prev; //ָ����һ��ѵ��ͼ���ָ��

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
		//Mat queryImg; //��ѯͼ��
		int trainImgNumbers; //ѵ������ͼ����Ŀ
		//ָ��ѵ��ͼ�񼯵�ָ�룬��˫������֯����,��ָ�������Ϊͷ��㲻�洢����
		TrainItem *trainingSet;

	public:
		Distance();
		~Distance();

		/* ����ѵ����
		 * @para fileName fileName�ļ��б��������е�ѵ��ͼ�������  
		 */
		void loadTrainingImgs(const string &fileName);

		/* ������룬�����ľ��밴��С��������
		 * @para queryImg ��ѯͼ��
		 * @para distanceType ����������ͣ�EUCLID_DISTANCE
		 */
		void computeDistance(const Mat &queryImg, int distanceType);

		/* ��õ�nС�ľ���
		 * @para n ��nС�ľ��룬�±��1��ʼ
		 * @return double ���ص�nС�ľ���
		 */
		double getDistance(int n) const;

		/* ��þ����nС��ͼ��ı�ţ���ż�����ͼ��ʱ��ͼ����
		 * @para n ��nС��ͼ��ı�ţ��±��1��ʼ
		 * @return string ���ص�nС��ͼ��ı��
		 */
		string getLabel(int n) const;

		/* ��ȡ��n��ѵ����ͼ��
		 * @para n ��n��ѵ����ͼ���±��1��ʼ
		 * @return Mat ѵ����ͼ��
		 */
		Mat getTrainingImg(int n) const;

		/* ��ȡ���ص�ѵ����ͼ����Ŀ
		 * @return int ����ѵ������ͼ�����Ŀ
		 */
		int getTrainingNum() const;

	private:
		/* ����sift������֮��ľ���
		 * @para trainSet ָ��˫��ѵ���������ָ��
		 * @para queryImg ��ѯͼ��
		 */
		void computeSiftDistance(TrainItem *trainSet, const Mat &queryImg);

		/* ��ȡѵ����ÿ��ͼƬ��SIFT������
		 * @para number ÿ��ͼ����ȡ����������
		 */
		void extractSiftFeature(int number);
	}; // end class Distance
}

#endif // end __REGONITION_H__




