# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.layers import *  # Affine, Relu, SoftmaxWithLoss, BatchNormalization 등을 포함해야 함
from common.gradient import numerical_gradient
from collections import OrderedDict

class MulLayerNet:
    
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01, dropout_ratio=0.2):
        # 가중치 초기화
        self.params = {}
        self.layers = OrderedDict()
        self.layer_count = 5  # 사용할 Affine 계층의 수
        self.dropout_ratio = dropout_ratio
        
        # 첫번째 Affine 계층과 Relu 계층, 그리고 Dropout 계층 초기화
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['BatchNorm1'] = BatchNormalization(gamma=np.ones(hidden_size), beta=np.zeros(hidden_size))
        self.layers['Relu1'] = Relu()
        self.layers['Dropout1'] = Dropout(self.dropout_ratio)
        
        # 중간 Affine 계층과 Relu 계층, 그리고 Dropout 계층 초기화
        for i in range(2, self.layer_count):
            self.params[f'W{i}'] = weight_init_std * np.random.randn(hidden_size, hidden_size)
            self.params[f'b{i}'] = np.zeros(hidden_size)
            self.layers[f'Affine{i}'] = Affine(self.params[f'W{i}'], self.params[f'b{i}'])
            self.layers[f'BatchNorm{i}'] = BatchNormalization(gamma=np.ones(hidden_size), beta=np.zeros(hidden_size))
            self.layers[f'Relu{i}'] = Relu()
            self.layers[f'Dropout{i}'] = Dropout(self.dropout_ratio)
        
        # 마지막 Affine 계층 초기화
        self.params[f'W{self.layer_count}'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params[f'b{self.layer_count}'] = np.zeros(output_size)
        self.layers[f'Affine{self.layer_count}'] = Affine(self.params[f'W{self.layer_count}'], self.params[f'b{self.layer_count}'])
        
        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
        
    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        for i in range(1, self.layer_count + 1):
            grads[f'W{i}'] = numerical_gradient(loss_W, self.params[f'W{i}'])
            grads[f'b{i}'] = numerical_gradient(loss_W, self.params[f'b{i}'])
        
        return grads
        
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
                
        for i in range(1, self.layer_count + 1):
            grads[f'W{i}'] = self.layers[f'Affine{i}'].dW
            grads[f'b{i}'] = self.layers[f'Affine{i}'].db



        return grads
    