import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from common.optimizer import AdaGrad  # Adam 최적화기를 사용
from MulLayerNet import MulLayerNet
import matplotlib.pyplot as plt

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = MulLayerNet(input_size=784, hidden_size=150, output_size=10, dropout_ratio=0)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 200
learning_rate = 0.019  # 사용자 지정 학습률

train_loss_list = []
train_acc_list = []
test_acc_list = []


iter_per_epoch = max(train_size / batch_size, 1)

optimizer = AdaGrad(lr=learning_rate)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 오차역전파법으로 기울기를 구한다.
    grad = network.gradient(x_batch, t_batch)
    
    # Adam 최적화기로 매개변수 갱신
    optimizer.update(network.params, grad)
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(round(train_acc*100, 4), round(test_acc*100, 4))

plt.figure()
plt.plot(train_loss_list)
plt.title('Training Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

plt.figure()
plt.plot(train_acc_list, label='Train Accuracy')
plt.plot(test_acc_list, label='Test Accuracy')
plt.title('Training and Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Logging
with open('accuracy_log.txt', 'w') as log_file:
    log_file.write('Epoch\tTrain Accuracy\tTest Accuracy\n')
    for epoch, (train_acc, test_acc) in enumerate(zip(train_acc_list, test_acc_list)):
        log_file.write(f'{epoch}\t{train_acc}\t{test_acc}\n')
