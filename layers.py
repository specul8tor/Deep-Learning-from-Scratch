import numpy as np

class Dense:
	def __init__(self,n_inputs,n_neurons):
		self.weights = np.random.rand(n_neurons, n_inputs)
		self.biases = np.zeros((n_neurons,1))

	def forward(self,inputs):
		self.outputs = np.dot(self.weights, inputs.T) + self.biases

	def backward(self,inputs):
		self.weights_prime = inputs
		self.bias_prime = np.eye(self.biases.shape[0])
		self.input_prime = self.weights
		self.prime = [self.weights_prime, self.bias_prime, self.input_prime]

class Sigmoid:
	def forward(self,inputs):
		self.outputs = (1/(1+np.exp(-inputs))).T
	def backward(self,inputs):
		self.prime = np.zeros((inputs.shape[0],inputs.shape[0],inputs.shape[1]))
		for b in range(inputs.shape[1]):
			self.prime[:,:,b] = np.exp(-inputs[:,b])/((1+np.exp(-inputs[:,b]))**2)*np.eye(inputs.shape[0])

class SoftMax:
	def forward(self,inputs):
		self.outputs = np.exp(inputs)/np.sum(np.exp(inputs), axis=0,keepdims=True)
	def backward(self,inputs,shape,batchNumber):
		inputs = inputs.T
		self.prime = np.zeros((batchNumber,shape,shape))
		for b in range(batchNumber):
			self.prime[b] = np.diagflat(inputs[b])/np.sum(np.exp(inputs[b])) - np.dot(inputs[b],inputs[b].T)

class CostMeanSquared:
	def forward(self,inputs,target,length):
		self.outputs = np.sum((inputs.T - target)**2)/length
	def backward(self,inputs,target,length):
		self.prime = 2*(inputs.T - target)/length