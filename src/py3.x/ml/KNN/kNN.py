#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Created on Sep 16, 2010
Update  on 2017-05-18
Author: Peter Harrington/羊三/小瑶
GitHub: https://github.com/apachecn/AiLearning
"""

# 导入科学计算包numpy和运算符模块operator
from numpy import *
import operator

dating_test_set_file = '/Users/mtdp/OneDrive/data/data-master/ml/KNN/datingTestSet2.txt'


def classify0(in_x, data_set, labels, k):
    """
    Desc:
        kNN 的分类函数
    Args:
        in_x -- 用于分类的输入向量/测试数据
        data_set -- 训练数据集的 features
        labels -- 训练数据集的 labels
        k -- 选择最近邻的数目
    Returns:
        sorted_class_count[0][0] -- 输入向量的预测分类 labels

    注意：labels元素数目和dataSet行数相同；程序使用欧式距离公式.

    预测数据所在分类可在输入下列命令
    kNN.classify0([0,0], group, labels, 3)
    """

    # -----------实现 classify0() 方法的第一种方式------------------------------
    # 1. 距离计算
    # 训练数据集合的行数
    data_set_size = data_set.shape[0]
    # tile生成和训练样本对应的矩阵，并与训练样本求差
    """
    tile: 列-3表示复制的行数， 行-1／2表示对inx的重复的次数

    In [8]: tile(inx, (3, 1))
    Out[8]:
    array([[1, 2, 3],
        [1, 2, 3],
        [1, 2, 3]])

    In [9]: tile(inx, (3, 2))
    Out[9]:
    array([[1, 2, 3, 1, 2, 3],
        [1, 2, 3, 1, 2, 3],
        [1, 2, 3, 1, 2, 3]])
    """
    diff_mat = tile(in_x, (data_set_size, 1)) - data_set
    """
    欧氏距离： 点到点之间的距离
       第一行： 同一个点 到 dataSet 的第一个点的距离。
       第二行： 同一个点 到 dataSet 的第二个点的距离。
       ...
       第N行： 同一个点 到 dataSet 的第N个点的距离。

    [[1,2,3],[1,2,3]]-[[1,2,3],[1,2,0]]
    (A1-A2)^2+(B1-B2)^2+(c1-c2)^2
    """
    # 取平方
    sq_diff_mat = diff_mat ** 2
    # 将矩阵的每一行相加
    sq_distances = sq_diff_mat.sum(axis=1)
    # 开方 欧式距离
    distances = sq_distances ** 0.5
    # 根据距离排序从小到大的排序，返回对应的索引位置
    # argsort() 是将x中的元素从小到大排列，提取其对应的index（索引），然后输出到y。
    # 例如：y=array([3,0,2,1,4,5]) 则，x[3]=1最小，所以y[0]=3;x[5]=5最大，所以y[5]=5。
    # print 'distances=', distances
    # 欧式距离从小到大的index
    sort_distances = distances.argsort()
    # print 'distances.argsort()=', sortedDistIndicies

    # 2. 选择距离最小的k个点
    class_count = {}
    for i in range(k):
        # 找到该样本的类型
        vote_label = labels[sort_distances[i]]
        # 在字典中将该类型加一
        # 字典的get方法
        # 如：list.get(k,d) 其中 get相当于一条if...else...语句,参数k在字典中，字典将返回list[k];如果参数k不在字典中则返回参数d,如果K在字典中则返回k对应的value值
        # l = {5:2,3:4}
        # print l.get(3,0)返回的值是4；
        # Print l.get（1,0）返回值是0；
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    # 3. 排序并返回出现最多的那个类型
    # 字典的 items() 方法，以列表返回可遍历的(键，值)元组数组。
    # 例如：dict = {'Name': 'Zara', 'Age': 7}   print "Value : %s" %  dict.items()   Value : [('Age', 7), ('Name', 'Zara')]
    # sorted 中的第2个参数 key=operator.itemgetter(1) 这个参数的意思是先比较第几个元素
    # 例如：a=[('b',2),('a',1),('c',0)]  b=sorted(a,key=operator.itemgetter(1)) >>>b=[('c',0),('a',1),('b',2)] 可以看到排序是按照后边的0,1,2进行排序的，而不是a,b,c
    # b=sorted(a,key=operator.itemgetter(0)) >>>b=[('a',1),('b',2),('c',0)] 这次比较的是前边的a,b,c而不是0,1,2
    # b=sorted(a,key=opertator.itemgetter(1,0)) >>>b=[('c',0),('a',1),('b',2)] 这个是先比较第2个元素，然后对第一个元素进行排序，形成多级排序。
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]

    # ------------------------------------------------------------------------------------------------------------------------------------------
    # 实现 classify0() 方法的第二种方式

    # """
    # 1. 计算距离

    # 欧氏距离： 点到点之间的距离
    #    第一行： 同一个点 到 dataSet的第一个点的距离。
    #    第二行： 同一个点 到 dataSet的第二个点的距离。
    #    ...
    #    第N行： 同一个点 到 dataSet的第N个点的距离。

    # [[1,2,3],[1,2,3]]-[[1,2,3],[1,2,0]]
    # (A1-A2)^2+(B1-B2)^2+(c1-c2)^2

    # inx - dataset 使用了numpy broadcasting，见 https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html
    # np.sum() 函数的使用见 https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.sum.html
    # """


#   dist = np.sum((inx - dataset)**2, axis=1)**0.5

# """
# 2. k个最近的标签

# 对距离排序使用numpy中的argsort函数， 见 https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.sort.html#numpy.sort
# 函数返回的是索引，因此取前k个索引使用[0 : k]
# 将这k个标签存在列表k_labels中
# """
# k_labels = [labels[index] for index in dist.argsort()[0 : k]]
# """
# 3. 出现次数最多的标签即为最终类别

# 使用collections.Counter可以统计各个标签的出现次数，most_common返回出现次数最多的标签tuple，例如[('lable1', 2)]，因此[0][0]可以取出标签值
# """
# label = Counter(k_labels).most_common(1)[0][0]
# return label

# ------------------------------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
def file2matrix(filename):
    """
    导入训练数据
    :param filename: 数据文件路径
    :return: 数据矩阵returnMat和对应的类别classLabelVector
    """
    fr = open(filename, 'r')
    # 获得文件中的数据行的行数
    number_of_line = len(fr.readlines())
    # 生成对应的空矩阵
    # 例如：zeros(2，3)就是生成一个 2*3 的矩阵，各个位置上全是 0 
    return_mat = zeros((number_of_line, 3))  # prepare matrix to return
    # prepare labels return
    class_label_vector = []
    fr = open(filename, 'r')
    index = 0
    for line in fr.readlines():
        # str.strip([chars]) --返回移除字符串头尾指定的字符生成的新字符串
        line = line.strip()
        # 以 '\t' 切割字符串
        list_from_line = line.split('\t')
        # 每列的属性数据，即 features
        return_mat[index] = list_from_line[0: 3]
        # 每列的类别数据，就是 label 标签数据
        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    # 返回数据矩阵returnMat和对应的类别classLabelVector
    return return_mat, class_label_vector


def auto_norm(date_set):
    """
    Desc：
        归一化特征值，消除属性之间量级不同导致的影响
    Args：
        dataSet -- 需要进行归一化处理的数据集
    Returns：
        normDataSet -- 归一化处理后得到的数据集
        ranges -- 归一化处理的范围
        minVals -- 最小值

    归一化公式：
        Y = (X-Xmin)/(Xmax-Xmin)
        其中的 min 和 max 分别是数据集中的最小特征值和最大特征值。该函数可以自动将数字特征值转化为0到1的区间。
    """
    # 计算每种属性的最大值、最小值、范围
    min_vals = date_set.min(0)
    max_vals = date_set.max(0)
    # 极差
    ranges = max_vals - min_vals
    # -------第一种实现方式---start-------------------------
    # norm_data_set = zeros(shape(dataSet))
    # m = dataSet.shape[0]
    # # 生成与最小值之差组成的矩阵
    # norm_data_set = dataSet - tile(min_vals, (m, 1))
    # # 将最小值之差除以范围组成矩阵
    # # element wise divide
    # norm_data_set = norm_data_set / tile(ranges, (m, 1))
    # -------第一种实现方式---end---------------------------------------------

    # # -------第二种实现方式---start---------------------------------------
    norm_data_set = (date_set - min_vals) / ranges
    # # -------第二种实现方式---end---------------------------------------------
    return norm_data_set, ranges, min_vals


def img2vector(filename):
    """
    Desc：
        将图像数据转换为向量
    Args：
        filename -- 图片文件 因为我们的输入数据的图片格式是 32 * 32的
    Returns:
        returnVect -- 图片文件处理完成后的一维矩阵

    该函数将图像转换为向量：该函数创建 1 * 1024 的NumPy数组，然后打开给定的文件，
    循环读出文件的前32行，并将每行的头32个字符值存储在NumPy数组中，最后返回数组。
    """
    return_vector = zeros((1, 1024))
    fr = open(filename, 'r')
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vector[0, 32 * i + j] = int(line_str[j])
    return return_vector

