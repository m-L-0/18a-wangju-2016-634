Tensorflow Schoolwork
==
任务一：将鸢尾花数据集安装8 : 2的比例划分成训练集与验证集
--
iris=datasets.load_iris()

x=iris.data#获取花卉两列数据集(二维矩阵150，4)

y=iris.target#一维向量,鸢尾花的类别

x_train,x_validation,y_train,y_validation=train_test_split(x,y,test_size=0.2,random_state=42)

任务二，三：设计模型及训练模型
--
###counter 函数的作用：返回数组中频率最高的一个数及出现次数

from collections import Counter

def counter(arr):
    
    return Counter(arr).most_common(1) # 返回出现频率最高的一个数

##k近邻模型的原理及代码实现：

  1.计算测试集每个样本与训练集每个样本的距离（我用的l1距离）
  
        distance=tf.reduce_sum(tf.abs(tf.add(x_train,tf.negative(x_validation[i]))),reduction_indices=1)
        
  2.根据距离度量得到每个测试集样本最近的前k个近邻的标签
  
  （先得到前k个近邻在训练集中的索引，在找到相应的类别标签）
  
       indices_ = tf.nn.top_k(-distance, k=k_num).indices#前k个近邻索引
        
        with tf.Session() as sess:
            
            sess.run(distance)
            
            sess.run(-value_)
            
            indices=np.array(sess.run(indices_))
        
        #获取前k个近邻的标签
        
        a=[]
        
        for j in range(len(indices)):
            
            a.append(y_train[indices[j]])
      
    
3.k个近邻中出现次数最多的类别预测为测试样本的类别

        #得到a中出现次数最多的一个元素，及出现次数
        
        z=counter(a)
        
        print('indices',indices,'a',a,'z',z)
        
        pred_class_label=z[0][0]#max(set(a))
        

任务四：验证模型
----

###得到分类正确率的代码：

        pred_class_label=z[0][0]#max(set(a))
        
        true_class_label = y_validation[i]
  
        #print("Test", i, "Predicted Class Label:", pred_class_label,
        
        #"True Class Label:",  true_class_label )
        
        if pred_class_label == true_class_label:
            
            accuracy += 1
    
        print('k',k_num,(accuracy/30)*100,'%')
        
     
     #验证模型

     for m in range(1,5):
    
          k_nn(m,Ntest)
         
###经过验证模型，k取值1到4 正确率都是100%，选择超参数k=1

任务五：提交模型
--------
已提交
      
