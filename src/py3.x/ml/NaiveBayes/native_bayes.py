import numpy as np


def create_watermelon_data_set():
    watermelon_data_set = [
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

    watermelon_labels = ['color', 'root', 'sound', 'texture', 'button', 'touch']
    return watermelon_data_set, watermelon_labels


class NaiveBayes:
    def __init__(self):
        # key 是类名
        # val 为字典
        # PClass 表示该类
        # PFeature 表示对应各个特征的概率
        self.model = {}

    def naive_bayes(self, data_set, labels):
        data_set_group_by, classifier_prob = group_by_data_set(data_set)
        print(classifier_prob)
        bayes_model = {}
        for className, prob in classifier_prob.items():
            bayes_model[className] = {'prob': prob, 'feature': {}}

        for className, group in data_set_group_by.items():
            for index, label in enumerate(labels):
                feature = [g[index] for g in group]
                prob = calculate_feature_prob(feature)
                bayes_model[className]['feature'][label] = prob
        self.model = bayes_model
        return bayes_model

    def classify(self, data):
        """
        P(C|F1F2...Fn) = P(F1F2...Fn|C)P(C)/P(F1F2...Fn)

        P(F1F2...Fn|C) P(C) = P(F1|C) * P(F2|C) * ... * P(Fn|C) * P(C) -> lg(P(F1|C)) + lg(P(F2|C)) + ... + lg(P(Fn|C)) + lg(P(C))
        :param data:
        :return:
        """
        current_max_rate = None
        current_select = None
        for name_class, bayes_model in self.model.items():
            rate = 0.0
            classify_prob = bayes_model['prob']
            rate += np.log10(classify_prob)
            feature_model = bayes_model['feature']

            for feature_key, feature_value in data.items():
                prob = feature_model.get(feature_key)
                if not prob:
                    continue
                p = prob.get(feature_value, 0)
                print(feature_key, '->', p)
                rate += np.log10(p)
            if current_max_rate == None or rate > current_max_rate:
                current_max_rate = rate
                current_select = name_class
        return current_select


def group_by_data_set(data_set):
    """
    将 data_set 按标签分类 并计算标签概率
    :param data_set: 训练集
    :return: 按标签分类 、标签概率
    """
    data_set_group_by = {}
    class_count = {}
    classifier_train_set_count = len(data_set)
    for data in data_set:
        classifier = data[-1]
        if classifier not in data_set_group_by.keys():
            class_count[classifier] = 1
            data_set_group_by[classifier] = []
        class_count[classifier] += 1
        data_set_group_by[classifier].append(data)
    keys_count = len(class_count.keys())
    classifier_prob = dict(
        zip(class_count, map(lambda count: count / (classifier_train_set_count + keys_count), class_count.values())))
    return data_set_group_by, classifier_prob


def calculate_feature_prob(feature_set):
    """
    将 data_set 按标签分类 并计算标签概率
    :param data_set: 训练集
    :return: 按标签分类 、标签概率
    """
    class_count = {}
    classifier_train_set_count = len(feature_set)
    for feature in feature_set:
        if feature not in class_count.keys():
            class_count[feature] = 1
        class_count[feature] += 1
    keys_count = len(class_count.keys())
    classifier_prob = dict(
        zip(class_count, map(lambda count: count / (classifier_train_set_count + keys_count), class_count.values())))
    return classifier_prob


if __name__ == '__main__':
    watermelon_data_set, watermelon_labels = create_watermelon_data_set()
    watermelon_labels = ['color', 'root', 'sound', 'texture', 'button', 'touch']

    test_data = {
        'color': 'green',
        'root': 'curl',
        'sound': 'dull',
        'texture': 'fuzzy',
        'button': 'concave',
        'touch': 'hard'
    }
    naiveBayes = NaiveBayes()
    return_bayes_model = naiveBayes.naive_bayes(watermelon_data_set, watermelon_labels)
    classifier = naiveBayes.classify(test_data)
    print(classifier)
    # group_by_data_set(watermelon_data_set)
