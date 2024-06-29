from sklearn.metrics.cluster import fowlkes_mallows_score, adjusted_rand_score

def cluster_performance(y_true, y_pred):
    '''
    返回FM指数和Rand指数
    :param y_true:参考模型的簇划分，类型为ndarray
    :param y_pred:聚类模型给出的簇划分，类型为ndarray
    :return: FM指数，Rand指数
    '''

    #********* Begin *********#
    FM=fowlkes_mallows_score(y_true,y_pred)
    Rand=adjusted_rand_score(y_true,y_pred)
    return FM,Rand
    #********* End *********#
