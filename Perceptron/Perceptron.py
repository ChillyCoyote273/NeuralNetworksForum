import random

class Perceptron():
	def __init__(self, size):
		self.weights = []
		for i in range(size):
			self.weights.append(random.uniform(-1, 1))
		self.bias = random.uniform(-1, 1)
	
	def forward(self, x):
		total = 0
		for i in range(len(x)):
			total += x[i] * self.weights[i]
		total += self.bias
		return total
	
	def backward(self, correct, x, learning_rate):
		y = self.forward(x)
		for i in range(len(x)):
			self.weights[i] += learning_rate * (correct - y) * x[i]
		self.bias += learning_rate * (correct - y)
	
	def learn(self, inputs, outputs, learning_rate):
		for inp, out in zip(inputs, outputs):
			self.backward(out, inp, learning_rate)
