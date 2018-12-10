Hyperspectral<br>
==
一.整理数据集<br>
--
所有类别的数据为一个数据集，相应标签为一个测试集<br>
x_shape=[1071,622,362,547,358,729,1841,445,949]<br>

label = [2,3,5,6,8,10,11,12,14]<br>

#创建数据集和类别标签合并九个类别为一个数据集<br>

data=sio.loadmat('class9-train/data2_train.mat')#返回值是字典<br>

key = 'data'+str(2)+'_train'<br>

x=data[key]#(6924, 200)<br>

y=np.ones(6924)<br>

sum=0<br>

for m in range(9):<br>
    
    for n in range(sum,sum+x_shape[m]):
        
        y[n]=label[m]
        
    sum=sum+x_shape[m]
    
#547,358,729,1841,445,949<br>
#print(train_y[1071+622+362+547+358+729+1841+445+949-1])<br>

for i in range(1,9):<br>
    
    data = sio.loadmat('class9-train/data'+str(label[i])+'_train.mat')
    
    key = 'data'+str(label[i])+'_train'
    
    x=np.append(x,data[key],axis=0)
    
#print(x.shape)#(6924, 200)<br>
#print(y.shape)#(6924,)<br>

二.数据标准化<br>
--
data_D=preprocessing.StandardScaler().fit_transform(x)

三.训练集测试集划分<br>
--
data_train, data_test, label_train, label_test = train_test_split(data_D,data_L,test_size=0.5)

四.模型训练与拟合<br>
--
clf = SVC(kernel='rbf',gamma=0.125,C=15) <br>

C_range = np.logspace(-2, 10, 13)# logspace(a,b,N)把10的a次方到10的b次方区间分成N份<br>

gamma_range = np.logspace(-9, 3, 13)<br>

param_grid = dict(gamma=gamma_range, C=C_range)<br>

grid = GridSearchCV(clf, param_grid)<br>

grid.fit(data_train,label_train) <br>

pred = grid.predict(data_test)<br>

五.计算正确率<br>
--
accuracy = metrics.accuracy_score(label_test, pred)*100 

六.存储结果学习模型，方便之后的调用<br>
--
joblib.dump(grid,'train_model.m')

七.对给定的数据集进行预测<br>
--
import joblib<br>

import scipy.io as scio<br>

data_test_final=scio.loadmat('class9-train/data_test_final.mat')<br>

data_test=data_test_final['data_test_final']<br>

clf = joblib.load("train_model.m")<br>

predict_label = clf.predict(data_test)<br>
