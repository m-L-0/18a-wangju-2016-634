任务一：将鸢尾花数据集画成图的形式。
=
1.借助sklearn包，导入数据集<br>
2.利用逻辑回归模型训练样本集<br>
3.绘制散点图<br>

任务二：确定一个合适的阈值，只有两个样本之间的相似度大于该阈值时，这两个样本之间才有一条边。<br>
任务三：求取带权邻接矩阵。<br/>
=
1.计算距离矩阵<br>
def euclidDistance(x1,x2,sqrt_flag=False):<br>
    res = np.sum((x1-x2)**2)  这么计算距离正好对应求高斯核的公式<br>
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

根据距离矩阵，计算每个点的前k个近邻，在对这个点与k个近邻的距离带入高斯核的公式，得到带权邻接矩阵。<br>
带权邻接矩阵中为0的代表两个点之间没有边，相反有边<br>
 def myKNN(S,k,sigma=0.187):<br>
    N=len(S)<br>
    A=np.zeros((N,N))<br>
    for i in range(N):<br>
        dist_with_index=zip(S[i],range(N))#带索引的距离<br>
        dist_with_index=sorted(dist_with_index,key=lambda x:x[0])#按距离从小到大排序<br>
        neighbours_id=[dist_with_index[m][1] for m in range(k+1)]#得到前k个最近邻的索引<br>
        # xi's k nearest neighbours<br>
        for j in neighbours_id:# xj is xi's neighbour<br>
            A[i][j] = np.exp(-S[i][j]/2/sigma/sigma)#高斯核<br>
            A[j][i] = A[i][j] # mutually 邻接矩阵是对称的<br>
    return A<br>
   
   任务四：根据邻接矩阵进行聚类。<br>
   =

谱聚类实现过程<br>
1.计算距离矩阵<br>
2.利用KNN计算邻接矩阵A<br>
3.由A计算度数矩阵D和拉普拉斯矩阵L<br>
#3标准化的拉普拉斯矩阵<br>
def calLaplacianMatrix(adjacentMatrix):<br>
    #compute the Degree Matrix:D=sum(A)<br>
    degreeMatrix=np.sum(adjacentMatrix,axis=1)<br>
    #compute the Laplacian Matrix:L=D-A<br>
    laplacianMatrix=np.diag(degreeMatrix)-adjacentMatrix#度数矩阵是对角矩阵<br>
    #normailize<br>
     # D^(-1/2) L D^(-1/2)<br>
    sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix ** (0.5)))<br>
    return np.dot(np.dot(sqrtDegreeMatrix, laplacianMatrix), sqrtDegreeMatrix)<br>

4.标准化L=D^(-1/2) L D^(-1/2)<br>
5.对称阵L=D^(-1/2) L D^(-1/2)进行特征值分解，得到特征向量Hnn#6.将Hnn当成样本送入Kmeans<br>
7.获得聚类结果C=（C1，C2，....Ck）<br>
