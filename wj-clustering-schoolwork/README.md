任务一：将鸢尾花数据集画成图的形式。
=
1.借助sklearn包，导入数据集<br>
2.利用逻辑回归模型训练样本集<br>
3.绘制散点图<br>

任务二：确定一个合适的阈值，只有两个样本之间的相似度大于该阈值时，这两个样本之间才有一条边。
=
1.计算距离矩阵<br>
def euclidDistance(x1,x2,sqrt_flag=False):<br>
    res = np.sum((x1-x2)**2)#这么计算距离正好对应求高斯核的公式<br>
    if sqrt_flag:<br>
        res = np.sqrt(res)<br>
    return res<br>
def calEuclidDistanceMatrix(X): <br>
    X = np.array(X)<br>
    S = np.zeros((len(X), len(X)))<br>
    for i in range(len(X)):<br>
        for j in range(i+1, len(X)): <br>
            S[i][j] = 1.0 * euclidDistance(X[i], X[j]) <br>
            S[j][i] = S[i][j] <br>
    return S<br>
2.利用knn计算邻接矩阵<br>
