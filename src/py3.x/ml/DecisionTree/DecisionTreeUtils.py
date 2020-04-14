from numpy import *


def create_tree(data_set, labels):
    """
    创建决策树
    :param data_set: 要创建决策树的训练数据集
    :param labels: 训练数据集中特征对应的含义的labels 不是目标变量
    :return: 创建完成的决策树
    """
    # 训练数据集的分类标签
    classifier_list = [classifier[-1] for classifier in data_set]
    # 如果数据集的分类标签只有一种，直接返回这个结果
    # 第一个停止条件 所有类标签完全相同，则直接返回该类标签
    if classifier_list.count(classifier_list[0]) == len(classifier_list):
        return classifier_list[0]

    # 如果数据集只有1列 那么最初出现label次数最多的一类，作为结果
    # 第二个停止条件 使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分
    if len(data_set[0]) == 1:
        return majority_count(classifier_list)

    # 选择最优列 得到最优列对应的标签含义
    best_feature_index = choose_best_feature(data_set)
    # 获取最优feature
    best_feature = labels[best_feature_index]
    # 初始化决策树
    decision_tree = {best_feature: {}}
    # 删除这个最屌的标签
    del labels[best_feature_index]
    # 取出最优列 然后对他的分支继续做分类
    feature_values = [data[best_feature_index] for data in data_set]
    unique_feature_values = set(feature_values)
    for value in unique_feature_values:
        sub_labels = labels[:]
        decision_tree[best_feature][value] = create_tree(split_data_set(data_set, best_feature_index, value),
                                                         sub_labels)
    return decision_tree


def majority_count(classifier_list):
    """
    选择出现次数最多的一个结果
    :param classifier_list:lable

    :return:
    """
    classifier_count = {}
    for vote in classifier_list:
        if vote not in classifier_count.keys():
            classifier_count[vote] = 0
        classifier_count[vote] += 1
    # 倒叙排列classifier_count得到一个字典集合，然后取出第一个就是结果，即出现次数最多的结果
    sorted_classifier_count = sorted(classifier_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_classifier_count[0][0]


def choose_best_feature(data_set):
    """
    选择切分数据集的最佳特征
    :param data_set: 需要切分的数据集合
    :return: 最优特征列
    """
    # 求feature 个数 最后一列是 label
    number_feature = len(data_set[0]) - 1
    # 计算根结点的 信息熵
    root_information_entropy = calculate_information_entropy(data_set)
    # 最优信息增益值，最优feature 编号
    best_info_gain, best_feature = 0.0, -1
    # 循环所有标签 计算最优的信息增益 整出一个最好的标签 就跟养蛊一样
    for i in range(number_feature):
        # 获取每一个实例的feature 组成list 然后去重计算feature每种属性的信息熵
        feature_list = [feature[i] for feature in data_set]
        unique_feature_list = set(feature_list)
        # 临时信息熵
        tem_entropy = 0.0
        # 遍历当前特征中所有唯一属性对每种属性划分一次数据集合，计算熵，然后对这个特征值的熵求和
        for feature in unique_feature_list:
            sub_data_set = split_data_set(data_set, i, feature)
            probability = len(sub_data_set) / float(len(data_set))
            tem_entropy += probability * calculate_information_entropy(sub_data_set)
        # Gain信息增益 划分数据集前后信息变化 获取信息熵最大的值
        information_gain = root_information_entropy - tem_entropy
        print('information_gain', information_gain, 'best_feature=', i, root_information_entropy, tem_entropy)
        if information_gain > best_info_gain:
            best_info_gain = information_gain
            best_feature = i
    return best_feature


def calculate_information_entropy(data_set):
    """
    计算给定数据集的信息熵 也叫香农熵
    :param data_set: 数据集合
    :return: 返回 每一组 feature 下的某个分类下，香农熵的信息期望
    """
    # 求参与数据训练集的数据量
    num_entries = len(data_set)
    # 计算分类标签 label 出现的次数
    label_counts = {}
    for feature_vector in data_set:
        # 每行数据最后一个数据代表标签
        current_label = feature_vector[-1]
        # 记录label 以及每个label出现的次数 为了算标签的概率
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1

    information_entropy = 0.0
    for label_count in label_counts:
        # 使用所有类标签的发生概率计算类别出现的概率
        probability = float(label_counts[label_count]) / num_entries
        # 计算信息熵
        information_entropy = information_entropy - (probability * log2(probability))
    return information_entropy


def split_data_set(data_set, index, value):
    """
    拆分数据集合 通过遍历 data_set 找到index对应的列并且值为value的所有行
    划分出新的数据集合
    :param data_set: 待划分数据集
    :param index: 数据集第几列，对应的是feature的index
    :param value: 对应特征的值
    :return: index列为value的数据集
    """
    return [data[:index] + data[index + 1:] for data in data_set for i, v in enumerate(data) if
            i == index and v == value]


def classify(decision_tree, feature_labels, test_vector):
    """
    对新数据进行分类
    :param decision_tree: 决策树
    :param feature_labels: feature 对应的标签名称
    :param test_vector: 测试数据
    :return: 分类结果 需要映射 label 才知道名字
    """
    # 获取根结点key值
    root_key = list(decision_tree.keys())[0]
    # 获取根结点对应的value值
    root_values = decision_tree[root_key]
    # 判断根结点名称 获取对应的label index 这样就知道如何根据输入 test_vector 对照树做分类
    feature_index = feature_labels.index(root_key)
    # 测试数据，找到根结点对应的label位置，也就知道从输入的数据的第几位来分类
    test_key = test_vector[feature_index]
    value_of_feature = root_values[test_key]
    print('---', root_key, '---', root_values, '---', test_key, '>>>', value_of_feature)
    # 判断分支是否结束，判断 value_of_feature 是否是 dict
    if isinstance(value_of_feature, dict):
        classifier_label = classify(value_of_feature, feature_labels, test_vector);
    else:
        classifier_label = value_of_feature
    return classifier_label



