from ml.DecisionTree import DecisionTreeUtils
from ml.DecisionTree import decisionTreePlot


def create_watermelon_data_set():
    data_set = [
        ['green', 'curl', 'ring', 'clear', 'sag', 'hard', 'yes'],  # 1
        ['black', 'curl', 'dull', 'clear', 'sag', 'hard', 'yes'],  # 2
        ['black', 'curl', 'ring', 'clear', 'sag', 'hard', 'yes'],  # 3
        ['green', 'curl', 'dull', 'clear', 'sag', 'hard', 'yes'],  # 4
        ['white', 'curl', 'ring', 'clear', 'sag', 'hard', 'yes'],  # 5
        ['green', 'little', 'ring', 'clear', 'concave', 'soft', 'yes'],  # 6
        ['black', 'little', 'ring', 'paste', 'concave', 'soft', 'yes'],  # 7
        ['black', 'little', 'ring', 'clear', 'concave', 'hard', 'yes'],  # 8
        ['black', 'little', 'dull', 'paste', 'concave', 'hard', 'no'],  # 9
        ['green', 'strong', 'crispy', 'clear', 'flat', 'soft', 'no'],  # 10
        ['white', 'strong', 'crispy', 'fuzzy', 'flat', 'hard', 'no'],  # 11
        ['white', 'curl', 'ring', 'fuzzy', 'flat', 'soft', 'no'],  # 12
        ['green', 'little', 'ring', 'paste', 'sag', 'hard', 'no'],  # 13
        ['white', 'little', 'dull', 'paste', 'sag', 'hard', 'no'],  # 14
        ['black', 'little', 'ring', 'clear', 'concave', 'soft', 'no'],  # 15
        ['white', 'curl', 'ring', 'fuzzy', 'flat', 'hard', 'no'],  # 16
        ['green', 'curl', 'dull', 'fuzzy', 'concave', 'hard', 'no']  # 17
    ]
    labels = ['color', 'root', 'sound', 'texture', 'button', 'touch']
    return data_set, labels


def create_data_set():
    """
    创建数据集
    :return:
    """
    data_set = [['1', '1', 'yes'],
                ['1', '1', 'yes'],
                ['1', '0', 'no'],
                ['0', '1', 'no'],
                ['0', '1', 'no']]

    labels = ['no surfacing', 'flippers']
    return data_set, labels


def test_fish():
    """
    Desc:
        对动物是否是鱼类分类的测试函数，并将结果使用 matplotlib 画出来
    """
    # 1.创建数据和结果标签
    data_set, labels = create_watermelon_data_set()
    print(data_set, labels)

    import copy
    decision_tree = DecisionTreeUtils.create_tree(data_set, copy.deepcopy(labels))
    print(decision_tree)
    # [1, 1]表示要取的分支上的节点位置，对应的结果值
    print("----")
    # print(DecisionTreeUtils.classify(decision_tree, labels, [1, 1]))
    decisionTreePlot.createPlot(decision_tree)


if __name__ == '__main__':
    test_fish()
