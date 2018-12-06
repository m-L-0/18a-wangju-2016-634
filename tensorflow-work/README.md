Tensorflow Schoolwork
==
任务一：将鸢尾花数据集安装8 : 2的比例划分成训练集与验证集
--
iris=datasets.load_iris()

x=iris.data#获取花卉两列数据集(二维矩阵150，4)

y=iris.target#一维向量,鸢尾花的类别

x_train,x_validation,y_train,y_validation=train_test_split(x,y,test_size=0.2,random_state=42)

