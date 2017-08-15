# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 16:21:46 2017

@author: zhihuazhang
"""
#本程序由UESTC的BigMoyan完成，并供所有人免费参考学习，但任何对本程序的使用必须包含这条声明
from input_data import input_data
import numpy as np
import math
########################
#读入数据
########################
sample, label, test_s, test_l = input_data() #灰度值
sample = np.array(sample,dtype='float')  #读入矩阵，类型转换
sample/= 256.0       # 特征  灰度值归一化
test_s = np.array(test_s,dtype='float') 
test_s /= 256.0
########################
#神经网络配置
########################
sample_num=len(sample)  #矩阵维数，样本总数
in_num=len(sample[0])  #sample矩阵列数，即单个样本的输入向量(数量=输入节点数)
out_num=10   #0~9，10类
hid_num=15   #隐藏节点数(经验公式) ?
w1 = 0.2*np.random.random((in_num, hid_num))- 0.1   # 初始化输入-隐含层权矩阵 ?
w2 = 0.2*np.random.random((hid_num, out_num))- 0.1   # 初始化隐-输出层权矩阵
hid_offset = np.zeros(hid_num)    #隐含层偏置向量   ?
out_offset = np.zeros(out_num)    #输出层偏置向量
inp_lrate = 0.2             # 输入层-隐含层权值学习率
hid_lrate = 0.2             # 隐层-输出层学权值习率
########################
#激活函数定义 （此处选用sigmoid函数）
########################
def get_act(x):
    act_vec = []
    for i in x:  #此处是i取遍x向量中元素，完成激活函数对向量的处理。
        act_vec.append(1/(1+math.exp(-i)))
    act_vec=np.array(act_vec)
    return act_vec

########################
#训练网络
########################
for count in range(0,sample_num):
    t_label = np.zeros(out_num) #输出标签
    t_label[label[count]]=1     #label是标签，字典用法 
   
    #前向过程
    hid_value = np.dot(sample[count],w1)+hid_offset #隐层dot是乘积和
    hid_act = get_act(hid_value)                #隐层激活值
    out_value = np.dot(hid_act, w2) + out_offset  # 输出层dot是乘积和
    out_act = get_act(out_value)                # 输出层激活值
   
    #后向过程
    e = t_label - out_act                     # 输出值与真值间的误差
    out_delta = e * out_act * (1-out_act)         # 输出层delta计算   这一步是公式中gi值
    hid_delta = hid_act * (1-hid_act) * np.dot(w2, out_delta) # 隐层delta计算  这一步是公式中eh值
    for i in range(0, out_num):#输出层
        w2[:,i] += hid_lrate * out_delta[i] * hid_act   # 更新隐层到输出层权向量
    for i in range(0, hid_num):#隐层
        w1[:,i] += inp_lrate * hid_delta[i] * sample[count]      # 更新输出层到隐层的权向量
   
    out_offset += hid_lrate * out_delta                             # 输出层偏置更新
    hid_offset += inp_lrate * hid_delta
print('Training Finished!')
########################
#训练数据_误差统计
########################
temp=0
for count in range(len(sample)):
    hid_value = np.dot(sample[count], w1) + hid_offset       # 隐层值
    hid_act = get_act(hid_value)                # 隐层激活值
    out_value = np.dot(hid_act, w2) + out_offset             # 输出层值
    out_act = get_act(out_value)                # 输出层激活值
    if np.argmax(out_act) == label[count]:
        temp+=1
print('Train_Set Error is: %.2f%%'%((1-float(temp)/len(sample))*100))
train_error=(float)((1-float(temp)/len(sample))*100)
###################################################################################################
# 测试数据误差统计
###################################################################################################
temp=0
for count in range(len(test_s)):
    hid_value = np.dot(test_s[count], w1) + hid_offset       # 隐层值
    hid_act = get_act(hid_value)                # 隐层激活值
    out_value = np.dot(hid_act, w2) + out_offset             # 输出层值
    out_act = get_act(out_value)                # 输出层激活值
    if np.argmax(out_act) == test_l[count]:
        temp+=1
print('Test_Set Error is: %.2f%%'%((1-float(temp)/len(test_s))*100))
test_error=(float)((1-float(temp)/len(test_s))*100)
###################################################################################################
# 保存网络模型
###################################################################################################
Network = open("MyNetWork.txt", 'w')
Network.write('60000训练数据集_误差统计：%.2f%%\n'%train_error)
Network.write('10000测试数据集_误差统计：%.2f%%\n'%test_error)
Network.write("输入层节点数\n")
Network.write(str(in_num))
Network.write('\n')
Network.write("隐层节点数\n")
Network.write(str(hid_num))
Network.write('\n')
Network.write("输出层节点数\n")
Network.write(str(out_num))
Network.write('\n')
Network.write("输入-隐层节权值学习率\n")
Network.write(str(inp_lrate)) 
Network.write('\n')      
Network.write("隐-输出层节权值学习率\n")
Network.write(str(hid_lrate)) 
Network.write('\n')      

Network.write("输入-隐层节权值\n")           
for i in w1:
    for j in i:
        Network.write(str(j))
        Network.write(' ')
    Network.write('\n')
Network.write("隐-输出层节权值\n")
for i in w2:
    for j in i:
        Network.write(str(j))
        Network.write(' ')
Network.write('\n')
Network.write("输入-隐层偏置\n")
for i in hid_offset:
    Network.write(str(i))
    Network.write(' ')
Network.write('\n')
Network.write("隐-输出层偏置\n")
for i in out_offset:
    Network.write(str(i))
    Network.write(' ')
Network.write('\n')
Network.close()






