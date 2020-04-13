from ml.KNN import kNN
from numpy import *
import os

filename = '/Users/mtdp/OneDrive/data/data-master/ml/2.KNN/datingTestSet2.txt'


def create_data_set():
    """
    Desc:
        创建数据集和标签
    Args:
    Returns:
        group -- 训练数据集的 features
        labels -- 训练数据集的 labels
    调用方式
    import kNN
    group, labels = kNN.createDataSet()
    """
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def test_case1():
    """
    第一个例子
    :return:
    """
    group, labels = create_data_set();

    print(str(group))
    print(str(labels))
    print(kNN.classify0([0.1, 0.1], group, labels, 3))


def test_file2matrix():
    return_mat, class_label_vector = kNN.file2matrix(filename)
    print(return_mat)
    print(class_label_vector)


def test_dating_class():
    """
    对约会网站的测试方法，并将分类错误的数量和分类错误率打出来
    :return:
    """
    # 设置测试数据集比例 训练数据集比例是 1-test_ratio
    test_ratio = 0.1
    # 从文件中加载数据
    dating_data_mat, dating_labels = kNN.file2matrix(filename)
    # 归一化数据
    dating_data_mat_norm, ranges, min_vals = kNN.auto_norm(dating_data_mat)
    # 数据总行数
    m = dating_data_mat_norm.shape[0]
    # 设置测试样本数量
    num_test_vecs = int(m * test_ratio)
    error_count = 0
    for i in range(num_test_vecs):
        # 对数据测试
        classifier_result = kNN.classify0(dating_data_mat_norm[i], dating_data_mat_norm[num_test_vecs: m],
                                          dating_labels[num_test_vecs: m], 3)
        print("the classifier came back with %d , the real answer is %d" % (classifier_result, dating_labels[i]))
        error_count += classifier_result != dating_labels[i]

    print("error count is %d , the total error rate is : %f" % (error_count, (error_count / num_test_vecs)))


training_digits_file = '/Users/mtdp/OneDrive/data/data-master/ml/2.KNN/trainingDigits'
test_digits_file = '/Users/mtdp/OneDrive/data/data-master/ml/2.KNN/testDigits'


def handwriting_class_test():
    """
    手写数字识别分类器 并将分类错误数和分类错误率打印出来
    :return:
    """
    # 1. 导入数据
    handwriting_labels = []

    training_file_list = os.listdir(training_digits_file)
    m = len(training_file_list)
    print(m)
    training_mat = zeros((m, 1024))
    # handwriting_labels 存储0～9的index位置，trainingMat 存放的每个位置对应的图片向量
    for i in range(m):
        training_digits_filename = training_file_list[i]
        file_str = training_digits_filename.split('.')[0]
        classifier_number = int(file_str.split('_')[0])
        handwriting_labels.append(classifier_number)
        # 将 32 * 32 矩阵 -> 1 * 1024 矩阵
        training_mat[i] = kNN.img2vector(training_digits_file + '/' + training_digits_filename)

    # 2. 导入测试数据
    test_file_list = os.listdir(test_digits_file)
    error_count = 0
    test_m = len(test_file_list)
    for i in range(test_m):
        test_digits_filename = test_file_list[i]
        file_str = test_digits_filename.split('.')[0]
        classifier_number = int(file_str.split('_')[0])
        vector_under_test = kNN.img2vector(training_digits_file + '/' + test_digits_filename)
        classifier_result = kNN.classify0(vector_under_test, training_mat, handwriting_labels, 6)
        print("the classifier came back with: %d, the real answer is: %d" % (classifier_result, classifier_number))
        if classifier_result != classifier_number:
            error_count += 1
            print("\n the filename is " + test_digits_filename)
    print("\n the total number of errors is: %d" % error_count)
    print("\n the total error rate is: %f" % (error_count / test_m))


if __name__ == '__main__':
    # test_case1()
    # test_dating_class()
    # handwriting_class_test()
    handwriting_class_test()
