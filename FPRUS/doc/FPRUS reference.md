# FPRUS 使用说明

&emsp;&emsp;带test的头文件（如：eigensketch_test.h）是相关算法的测试文件，可供参考。
&emsp;&emsp;项目下的Python脚本extract_all_filename_in_directory.py用来提取指定路径
文件夹下的所有文件名到一个文本文件中，文件名附带相对于当前项目（即相对于脚本所在的
位置）的路径。如：../DataSet/CUHK_testing_cropped/photos/f-039-01.jpg

## class DataSet(transformation.h)

&emsp;&emsp;描述数据集的一个类，以后可能会继承该类(在写完Eigensketch类，才想到应该抽象出
一个基类)。

------------------

&emsp;&emsp;**构造器**

`DataSet(int imgNum)`
&emsp;&emsp;imgNum：数据集的大小


&emsp;&emsp;**枚举常量**

`FLAG_PHOTO = 0`
&emsp;&emsp;训练相片集的标志

`FLAG_SKETCH = 1`
&emsp;&emsp;训练素描集的标志

&emsp;&emsp;**成员函数**

`void loadTrainingPhotos(const string &photoFilename)`
&emsp;&emsp;加载训练相片集。
&emsp;&emsp;photoFilename：保存了训练相片集中的所有相片的相对路径与图像名。

`void releaseTrainingPhotos()`
&emsp;&emsp;释放训练相片集的内存。

`void loadTrainingSketches(const string &sketchFilename)`
&emsp;&emsp;加载训练素描集。
&emsp;&emsp;sketchFilename：保存了训练素描集中的所有素描的相对路径与图像名。

`void releaseTrainingSketches()`
&emsp;&emsp;释放训练素描集的内存。

`void loadTrainingImgs(const string &photoFilename, const string &sketchFilname)`
&emsp;&emsp;加载训练相片集与训练素描集。
&emsp;&emsp;photoFilename：保存了训练相片集中的所有相片的相对路径与图像名。
&emsp;&emsp;sketchFilename：保存了训练素描集中的所有素描的相对路径与图像名。

`void releaseTrainingImgs()`
&emsp;&emsp;释放训练相片集与训练素描集的内存。

`int getRows()`
&emsp;&emsp;得到训练相片集中的图像的像素行数。

`int getCols()`
&emsp;&emsp;得到训练相片集中的图像的像素列数。

`int getPixelNums()`
&emsp;&emsp;得到训练相片集中的图像的像素数。

`virtural Mat computeMean(int flag)`
&emsp;&emsp;计算训练集的均值。
&emsp;&emsp;flag：训练相片集或训练素描集的标志。FLAG_PHOTO, FLAG_SKETCH

`virtual void matrixToColVector(Mat &matVector, int flag)`
&emsp;&emsp;将训练集中的每张图像的像素矩阵拉伸为像素矢量（按行拉伸）。
&emsp;&emsp;matVector：矢量化后的结果。
&emsp;&emsp;flag：训练相片集或训练素描集的标志。FLAG_PHOTO, FLAG_SKETCH

`static void computeMean(const Mat *matSet, Mat &mean, int matNum)`
&emsp;&emsp;计算指定数据集的均值。
&emsp;&emsp;matSet：指定的数据集。
&emsp;&emsp;mean：计算的均值结果。
&emsp;&emsp;matNum：数据集大小。

`static void matrixToColVector(const Mat &matrix, Mat &matVector)`
&emsp;&emsp;将指定的矩阵拉伸为列矢量（按行拉伸）。
&emsp;&emsp;matrix：指定的矩阵。
&emsp;&emsp;matVector：矢量化的结果。

`static void colVectorToMatrix(const Mat &matVector, Mat &matrix)`
&emsp;&emsp;将指定的列矢量转化为矩阵的形式（按行转化）。
&emsp;&emsp;matVector：待转换为矩阵列矢量。
&emsp;&emsp;matrix：矩阵化的结果。


## class Eigensketch(transformation.h)

&emsp;&emsp;特征素描（Eigensketch）算法生成图像的重构素描图。使用该类的一个通用流程如下：
&emsp;&emsp;加载训练集：包括训练相片集与训练素描集，并且要求训练集中的每张图像有相同的大小。
&emsp;&emsp;计算参数：计算用于重构相片与素描所需的相关的参数。
&emsp;&emsp;重构图像：重构相片或素描图，要求待重构的相片的大小与训练中图像的大小相同。
&emsp;&emsp;释放训练集的内存。

--------------------

&emsp;&emsp;**构造器**

`Eigensketch(int imgNumber)`
&emsp;&emsp;imgNumber：数据集的大小

-------------------------------------

&emsp;&emsp;**成员函数**

`void loadTrainingPhoto(const string &photoFilename)`
&emsp;&emsp;加载训练集相片
&emsp;&emsp;photoFilename：保存了所有训练集相片的图像名的文件名，要求每行只有一个图像名。

`void loadTrainingSketch(const string &sketchFilename)`
&emsp;&emsp;加载训练集素描
&emsp;&emsp;sketchFilename：保存了所有训练集素描图的图像名的文件名，要求每行只有一个图像名。

`void loadTrainingImg(const string &photoFilename, const string &sketchFilename)`
&emsp;&emsp;加载训练集相片与素描
&emsp;&emsp;photoFilename：保存了所有训练集相片的图像名的文件名，要求每行只有一个图像名。
&emsp;&emsp;sketchFilename：保存了所有训练集素描图的图像名的文件名，要求每行只有一个图像名。

`void releaseTrainingPhoto()`
&emsp;&emsp;释放训练相片集的内存。

`void releaseTrainingSketch()`
&emsp;&emsp;释放训练素描集的内存。

`void releaseTrainingImg()`
&emsp;&emsp;释放训练相片集与素描集的内存。

`void computeParameters()`
&emsp;&emsp;计算重构相片与素描所需的相关系数。

`Mat reconstructPhoto(Mat &photo)`
&emsp;&emsp;重构相片
&emsp;&emsp;photo：待重构的相片
&emsp;&emsp;Mat：返回重构的相片

`Mat reconstructSketch(Mat &photo)`
&emsp;&emsp;重构素描
&emsp;&emsp;photo：待重构的相片
&emsp;&emsp;Mat：返回重构的素描


## class Distance(recognition.n)

&emsp;&emsp;计算图像或图像特征之间的距离，用于匹配。该类的使用流程如下：
&emsp;&emsp;加载训练集：用于测试的训练集（测试集）。
&emsp;&emsp;计算距离：支持Euclidean Distance，siftDistance等，参见函数说明。
&emsp;&emsp;得到相关的结果，参见具体函数。
&emsp;&emsp;注：计算基于特征点的距离待改进，每测试一张图像就会计算一遍测试集中每张图像
&emsp;&emsp;的特征点，重复了n次相同的操作，但这里不是特别最求速度，也就没改了。

-----------------

&emsp;&emsp;**构造器**
`Distance()`


&emsp;&emsp;**常量值**

`const static int EUCLID_DISTANCE = 0`
&emsp;&emsp;Euclidean距离。
`const static int SIFT_DISTANCE = 1`
&emsp;&emsp;SIFT距离。


&emsp;&emsp;**成员函数**

`void loadTrainingImgs(const string &fileName)`
&emsp;&emsp;载入训练集图像，该训练集图像是指测试集。
&emsp;&emsp;fileName：保存了所有图像名的文件。

`void computeDistance(const Mat &queryImg, int distanceType)`
&emsp;&emsp;计算查询图像与训练集图像中的每张图像的距离（或特征距离），并按从小到大的顺序排序。
&emsp;&emsp;queryImg：查询图像。
&emsp;&emsp;distanceType：EUCLID_DISTANCE、SIFT_DISTANCE

`double getDistance(int n) const`
&emsp;&emsp;获取计算得到的距离。
&emsp;&emsp;n：第n个最小距离，n从1开始。

`string getLabel(int n) const`
&emsp;&emsp;获取对应距离的标号，默认是图像名。
&emsp;&emsp;n：第n个最小距离对应的标号，n从1开始。

`Mat getTrainingImg(int n) const`
&emsp;&emsp;获取对应距离的图像。
&emsp;&emsp;n：第n个最小距离对应的图像，n从1开始。

`int getTrainingNum() const`
&emsp;&emsp;获取加载的训练集的图像数目。


