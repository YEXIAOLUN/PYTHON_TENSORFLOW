import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import executing

mnist=input_data.read_data_sets('mnist_data',one_hot=True)
#one_hot 独热码的编码（encoding)形式


#None表示张量(Tensor)的第一个维度可以是任何长度
input_x = tf.placeholder(tf.float32,[None,28*28])/255
output_y = tf.placeholder(tf.int32,[None,10])
input_x_image = tf.reshape(input_x,[-1,28,28,1])

test_x = mnist.test.images[:3000]
test_y = mnist.test.labels[:3000]

conv1=tf.layers.conv2d(inputs=input_x_image,
                       filters=32,
                       kernel_size=[5,5],
                       strides=1,
                       padding='same',
                       activation=tf.nn.relu)

pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2,2],
        strides=2)

conv2 = tf.layers.conv2d(inputs=pool1,   #形状[14,14,32]
                         filters=64,    #64个过滤器，输出的深度（depth）是64
                         kernel_size=[5,5], #过滤器的二维大小是（5*5）
                         strides=1,     #步长是1
                         padding='same',#same表示输出的大小不变，因此需要在外围补0 2圈
                         activation=tf.nn.relu) #激活函数是relu

#第 2 层池化（亚采样）
pool2 = tf.layers.max_pooling2d(
        inputs=conv2,   #形状[14,14,64]
        pool_size=[2,2],    #过滤器在二维的大小是(2*2)
        strides=2)      #步长是2
#形状[7,7,64]

#扁平化（flat）
flat=tf.reshape(pool2,[-1,7*7*64])  #形状[7*7*64]

#1024个神经元的全连接层
dense=tf.layers.dense(inputs=flat,units=1024,activation=tf.nn.relu)


#Dropout:丢弃50%,rate=0.5
dropout=tf.layers.dropout(inputs=dense,rate=0.5)

#10个神经元的全连接层
logits=tf.layers.dense(inputs=dropout,units=10) #输入，形状1*1*10


#计算误差（计算 Cross entropy (交叉熵),再用softmax 计算百分比概率
loss=tf.losses.softmax_cross_entropy(onehot_labels=output_y,logits=logits)

#Adam优化器来最小化误差,学习率0.001
train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)


#精度。计算预测值和实际标签的匹配程度
#返回(accuracy,update_op),会创建两个局部变量
accuracy = tf.metrics.accuracy(
    labels=tf.argmax(output_y, axis=1),
    predictions=tf.argmax(logits, axis=1),)[1]

#创建会话
sess=tf.Session()
#初始化变量：全局和局部
init=tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

node = executing.Source.executing(frame).node
print(node)
sess.run(init)
'''
for i in range(20000):
    batch=mnist.train.next_batch(50)  #从Train(训练)数据集里取 下一个50个样本
    train_loss,train_op_ = sess.run([loss,train_op],{input_x:batch[0],output_y:batch[1]})
    if i % 100 == 0:
        test_accuracy=sess.run(accuracy,{input_x:test_x,output_y:test_y})
        print('step=%d,train loss=%.4f,[test accuracy=%.2f]' % i,train_loss,test_accuracy)
'''
for i in range(5000):
    batch = mnist.train.next_batch(50)  # 从 Train（训练）数据集里取 “下一个” 50 个样本
    train_loss, train_op_ = sess.run([loss, train_op], {input_x: batch[0], output_y: batch[1]})
    if i % 100 == 0:
        test_accuracy = sess.run(accuracy, {input_x: test_x, output_y: test_y})
        print("第 {} 步的 训练损失={:.4f}, 测试精度={:.2f}".format(i, train_loss, test_accuracy))
#测试：打印20个预测值和真实值的对
test_output=sess.run(logits,{input_x:test_x[:20]})
inferenced_y=np.argmax(test_output,1)
print(inferenced_y,'inferenced numbers')
print(np.argmax(test_y[:20],1),'real numbers')

