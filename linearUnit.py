#coding:utf-8
from sensor import Perceptron
import numpy as np 



class LinearUnit(Perceptron):
    def __init__(self, input_num, activator):
        Perceptron.__init__(self, input_num, activator)

#定义激活函数
def f(x):
    return x;

def get_training_dataset():
    '''
    构建训练数据
    '''
    input_vecs = [[5], [3], [8], [1.4], [10.1]]
    #期望的输出
    labels = [5500, 2300, 7600, 1800, 11400]
    return input_vecs,labels

def train_linear_unit():
    '''
    使用线性单元
    '''
    #创建感知器，输入参数个数为1,激活函数为f
    p = LinearUnit(1, f)
    #训练，迭代10轮，学习速率为0.01
    input_vecs,labels = get_training_dataset()
    p.train(input_vecs, labels, 10, 0.01)
    return p

if __name__ == '__main__':
    #训练线性单元,获取权重
    linear_unit = train_linear_unit()
    print linear_unit
    #测试
    
    print '3.4, %.2f' % linear_unit.predict([3.4])
    print '15, %.2f' % linear_unit.predict([15])
    print '1.5, %.2f' % linear_unit.predict([1.5])
    print '6.3, %.2f' % linear_unit.predict([6.3])
    