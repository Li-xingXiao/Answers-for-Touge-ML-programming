
#encoding=utf8
import numpy as np
class smo:
    def __init__(self, max_iter=100, kernel='linear'):
        '''
        input:max_iter(int):最大训练轮数
              kernel(str):核函数，等于'linear'表示线性，等于'poly'表示多项式
        '''
        self.max_iter = max_iter
        self._kernel = kernel
    #初始化模型
    def init_args(self, features, labels):
        self.m, self.n = features.shape
        self.X = features
        self.Y = labels
        self.b = 0.0
        # 将Ei保存在一个列表里
        self.alpha = np.ones(self.m)
        self.E = [self._E(i) for i in range(self.m)]
        # 错误惩罚参数
        self.C = 1.0
    #********* Begin *********#    
    #kkt条件    
    def _KKT(self, i):
        y_g = self._g(i) * self.Y[i]
        if self.alpha[i] == 0:
            return y_g >= 1
        elif 0 < self.alpha[i] < self.C:
            return y_g == 1
        else:
            return y_g <= 1
        

    # g(x)预测值，输入xi（X[i]）
    def _g(self, i):
        r = self.b
        for j in range(self.m):
            r += self.alpha[j] * self.Y[j] * self.kernel(self.X[i], self.X[j])
        return r
 
    # 核函数,多项式添加二次项即可
    def kernel(self, x1, x2):
        if self._kernel == 'linear':
            return sum([x1[k] * x2[k] for k in range(self.n)])
        elif self._kernel == 'poly':
            return (sum([x1[k] * x2[k] for k in range(self.n)]) + 1) ** 2
        elif self._kernel == 'rbf':
            return np.exp(-sum([x1[k] - x2[k] for k in range(self.n)]) ** 2 / 2)
        else:
            return 0

    # E（x）为g(x)对输入x的预测值和y的差
    def _E(self, i):
        return self._g(i) - self.Y[i]

    #初始alpha
    def _init_alpha(self):
        # 外层循环首先遍历所有满足0<a<C的样本点，检验是否满足KKT
        index_list = [i for i in range(self.m) if 0 < self.alpha[i] < self.C]


        # 否则遍历整个训练集
        non_satisfy_list = [i for i in range(self.m) if i not in index_list]
        index_list.extend(non_satisfy_list)


    #选择alpha参数   
    def _select_alpha(self):
        # 外层循环首先遍历所有满足0<a<C的样本点，检验是否满足KKT
        index_list = [i for i in range(self.m) if 0 < self.alpha[i] < self.C]
        # 否则遍历整个训练集
        non_satisfy_list = [i for i in range(self.m) if i not in index_list]
        index_list.extend(non_satisfy_list)
        for i in index_list:
            if self._KKT(i):
                continue
            E1 = self.E[i]
            # 如果E2是+，选择最小的；如果E2是负的，选择最大的
            if E1 >= 0:
                j = min(range(self.m), key=lambda x: self.E[x])
            else:
                j = max(range(self.m), key=lambda x: self.E[x])
            return i, j
    #更新alpha
    def _update_alpha(self, i, j):
        # 计算上下界
        if self.Y[i] == self.Y[j]:
            L = max(0, self.alpha[i] + self.alpha[j] - self.C)
            H = min(self.C, self.alpha[i] + self.alpha[j])
        else:
            L = max(0, self.alpha[j] - self.alpha[i])
            H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
        # 计算alpha2的上下界
        
        E1 = self.E[i]
        E2 = self.E[j]
        # eta=K11+K22-2K12
        eta = self.kernel(self.X[i], self.X[j]) + self.kernel(
                self.X[j],
                self.X[j]) - 2 * self.kernel(self.X[i], self.X[j])
        if eta <= 0:
            return 1
        a = self.alpha[j] + self.Y[j] * (E2 - E1) / eta
        if a>H:
            alpha_j_new=H
        elif a<L:
            alpha_j_new=L
        else:
            alpha_j_new=a
        alpha_i_new = self.alpha[i] + self.Y[i] * self.Y[j] * (
                self.alpha[j] - alpha_j_new)
        b1_new = -E1 - self.Y[i] * self.kernel(self.X[i], self.X[i]) * (
alpha_i_new - self.alpha[i]) - self.Y[j] * self.kernel(self.X[j],self.X[i]) * (alpha_j_new - self.alpha[j]) + self.b
        b2_new = -E2 - self.Y[i] * self.kernel(self.X[i], self.X[j]) * (
alpha_i_new - self.alpha[i]) - self.Y[j] * self.kernel(self.X[j],self.X[j]) * (alpha_j_new - self.alpha[j]) + self.b
        if 0 < alpha_i_new < self.C:
            b_new = b1_new
        elif 0 < alpha_j_new < self.C:
            b_new = b2_new
        else:
            b_new = (b1_new + b2_new) / 2
        # 更新参数
        self.alpha[i] = alpha_i_new
        self.alpha[j] = alpha_j_new
        self.b = b_new
        # 更新E
        self.E[i] = self._E(i)
        self.E[j] = self._E(j)
        return 1
    

    #训练
    def fit(self, features, labels):
        '''
        input:features(ndarray):特征
              label(ndarray):标签
        '''
        self.init_args(features, labels)
        for t in range(self.max_iter):
            i, j = self._select_alpha()
            if self._update_alpha(i, j) == 0:
                break
            

      
    def predict(self, data):
        '''
        input:data(ndarray):单个样本
        output:预测为正样本返回+1，负样本返回-1
        '''
        r = self.b
        for i in range(self.m):
            r += self.alpha[i] * self.Y[i] * self.kernel(data, self.X[i])
        return 1 if r > 0 else -1
    

    #********* End *********# 

        
