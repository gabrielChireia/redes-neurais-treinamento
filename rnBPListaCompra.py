import numpy as np
import matplotlib.pyplot as plt

def sigmoid(sop):
    return 1.0/(1+np.exp(-1*sop))

def error(predicted, target):
    return np.power(predicted-target, 2)

def error_predicted_deriv(predicted, target):
    return 2*(predicted-target)

def sigmoid_sop_deriv(sop):
    return sigmoid(sop)*(1.0-sigmoid(sop))

def sop_w_deriv(x):
    return x

def update_w(w, grad, learning_rate):
    return w - learning_rate*grad

dataset = [1,5,6,0,7,2,4,8,3,6,3,6,2,3,3,3,2,6,4,6,1,2,8,3,4,4,2,2,5,1,6,6,2,8,3,6,1,5,8,3,5,1,7,4,7,3,2,5,5,1,6,4,4,2,7,5,8,2,0,3,6,3,8,6,5,5,6,7,7,1,5,6,1,6,5,2,4,2,4,7,1,6,3,1,6,0,0,5,1,3,6,8,1,0,3,1,2,6,8,3,1,4,5,3,0,6,4,4,5,0,3,5,0,4,6,2,2,1,4,4,3,4,8,7,4,4,1,0,1,2,2,2,2,3,3,5,7,1,4,5,7,6,3,1,1,2,1,1,7,1,0,8,4,6,3,2,6,2,5,5,0,4,8,8,7,0,8,4,3,7,8,1,4,0,5,3,1,3,5,5,7,3,6,8,8,4,6,5,2,7,5,5,7,5,1,8,2,5,8,5,0,1,2,7,8,5,1,4,6,4,3,1,0,1,0,8,2,5,8,6,1,2,5,4,5,5,4,3,4,3,3,8,7,6,5,3,6,7,8,6,0,5,4,8,5,2,2,6,7,2,7,3,8,7,1,6,5,0,5,7,6,0,8,8,0,0,7,8,4,8,8,5,0,4,3,0,0,0,2,7,4,0,2,7,2,0,1,1,5,7,8,8,7,3,8,2,4,5,7,5,0,0,3,7,1,0,0,1,0,5,7,6,2]

target = 4
learning_rate = 0.01
w1=numpy.random.rand()
w2=numpy.random.rand()

predicted_output = []

old_err = 0

for j in range(len(dataset)):
  y = dataset[j]
  predicted = sigmoid(y)
  err = error(predicted, target)

  predicted_output.append(predicted)

  g1 = error_predicted_deriv(predicted, target)
  g2 = sigmoid_sop_deriv(y)

  g3w1 = sop_w_deriv(dataset[j])
  g3w2 = sop_w_deriv(dataset[j])
    
  gradw1 = g3w1*g2*g1
  gradw2 = g3w2*g2*g1

  w1 = update_w(w1, gradw1, learning_rate)
  w2 = update_w(w2, gradw2, learning_rate)

plt.figure()
plt.plot(predicted_output)
plt.xlabel("Consumo")
plt.ylabel("Preco")