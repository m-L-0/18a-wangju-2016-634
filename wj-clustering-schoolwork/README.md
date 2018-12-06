#Clustering Schoolwork
==
任务一：将鸢尾花数据集画成图的形式。<br>
--
<br>
1.借助sklearn包，导入数据集<br>
2.利用逻辑回归模型训练样本集<br>
3.绘制散点图<br>

任务二：确定一个合适的阈值，只有两个样本之间的相似度大于该阈值时，这两个样本之间才有一条边。<br>
--
任务三：求取带权邻接矩阵。<br/>
--
###1.计算距离矩阵<br>
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
            S[j][i] = S[i][j] 距离矩阵是对称阵<br>
    return S<br>
    
###2.利用knn计算邻接矩阵<br>

根据距离矩阵，计算每个点的前k个近邻，在对这个点与k个近邻的距离带入高斯核的公式，得到带权邻接矩阵。<br>
带权邻接矩阵中为0的代表两个点之间没有边，相反有边<br>
sigma在0.1到10之间试的<br>
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
   --

谱聚类实现过程<br>
11.计算距离矩阵<br>
22.利用KNN计算邻接矩阵A<br>
33.由A计算度数矩阵D和拉普拉斯矩阵L<br>
44.标准化L=D^(-1/2) L D^(-1/2)<br>
55.对称阵L=D^(-1/2) L D^(-1/2)进行特征值分解，得到特征向量Hnn<br>
66.将Hnn当成样本送入Kmeans<br>
77.获得聚类结果C=（C1，C2，....Ck）<br>

###3标准化的拉普拉斯矩阵(以上两步，前两个任务就完成了)<br>

def calLaplacianMatrix(adjacentMatrix):<br>
    #compute the Degree Matrix:D=sum(A)<br>
    degreeMatrix=np.sum(adjacentMatrix,axis=1)<br>
    #compute the Laplacian Matrix:L=D-A<br>
    laplacianMatrix=np.diag(degreeMatrix)-adjacentMatrix#度数矩阵是对角矩阵<br>
    #normailize<br>
     # D^(-1/2) L D^(-1/2)<br>
    sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix ** (0.5)))<br>
    return np.dot(np.dot(sqrtDegreeMatrix, laplacianMatrix), sqrtDegreeMatrix)<br>

###5.对称阵L=D^(-1/2) L D^(-1/2)进行特征值分解，得到特征向量Hnn<br>
###6.将Hnn当成样本送入Kmeans<br>

eig_x,V= np.linalg.eig(Laplacian) # H'shape is n*n# H'shape is n*n<br>
eig_x=np.array(eig_x,dtype=np.float32)<br>
V=np.array(V,dtype=np.float32)<br>
eig_x=zip(eig_x,range(len(x)))<br>
eig_x=sorted(eig_x, key=lambda eig_x:eig_x[0])特征值从大到小排列<br>
H = np.vstack([V[:,i] for (v, i) in eig_x[:6]]).T#(shape (150,6))<br>
#试了试是前6个正确率最高<br>
#将H按行进行归一化<br>
Hmax,Hmin=H.max(axis=0),H.min(axis=0)<br>
H=(H-Hmin)/(Hmax-Hmin)<br>


7.获得聚类结果C=（C1，C2，....Ck）<br>

sp_kmeans=KMeans(n_clusters=3,n_init=13,random_state=17).fit(H)<br>

任务五：将聚类结果可视化，重新转换成图的形式，其中每一个簇应该用一种形状表示，比如分别用圆圈、三角和矩阵表示各个簇<br>
--
N=nx.Graph()<br>
for i in range(150):<br>
    N.add_node(i)<br>
#print(N.nodes())<br>
edglist=[]<br>
根据邻接矩阵，a[i][j]>0说明两个点之间有边
for i in range(150):<br>
    for j in range(150):<br>
        if(A[i][j]>0):<br>
            edglist.append((i,j))<br>
#print(edglist)<br>
三类对应三种不同的颜色
colorlist=[]<br>
N.add_edges_from(edglist)<br>
for i in range(150):<br>
    if(y[i]==0):<br>
        colorlist.append('r')<br>
    elif(y[i]==1):<br>
        colorlist.append('b')<br>
    else:<br>
        colorlist.append('orange')<br>
#print(N.neighbors((data_new[1][0],data_new[1][1])))<br>
#print(N[(data_new[1][0],data_new[1][1])])<br>
nx.draw(N,pos = nx.circular_layout(N),node_color = colorlist,edge_color = 'black',with_labels = False,font_size =5,node_shape='o' ,node_size =25,width=0.3)<br>

plt.show()<br>

任务六：求得分簇正确率<br>
--
#计算正确率<br>
count=0<br>
for i in range(150):<br>
    if y[i]==sp_kmeans.labels_[i]:<br>
        count=count+1<br>
print('Accuracy:',100*(count/150),'%')<br>

目前得到的正确率为81.3333333%

任务七：完成代码的描述文档
--
已完成
