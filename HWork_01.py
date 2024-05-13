import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from common.optimizer import AdaGrad
from MulLayerNet import MulLayerNet

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = MulLayerNet(input_size=784, hidden_size=15, output_size=10, dropout_ratio=0.01)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 1000
learning_rate = 0.019

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 오차역전파법으로 기울기를 구한다.
    grad = network.gradient(x_batch, t_batch)
    
    # 갱신
    # 여기서는 모든 계층의 가중치와 편향을 갱신해야 합니다.
    # 예를 들어, 가중치 키가 'W1', 'W2', ..., 'W100', 편향 키가 'b1', 'b2', ..., 'b100' 이라고 가정합니다.
    for key in network.params.keys():  # 모든 매개변수를 갱신
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(str(int(i//iter_per_epoch)+1)+'/'+str(int(iters_num//iter_per_epoch)+1), round(train_acc*100, 4), round(test_acc*100, 4))

