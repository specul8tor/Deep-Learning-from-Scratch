from preprocess import extract_raw, raw_2_numpy, data_batches
from layers import Dense, Sigmoid, SoftMax, CostMeanSquared
import numpy as np

### GLOBALS ###
BATCH_SIZE = 50
EPOCHS = 1000
LEARNING_RATE = 0.001


if __name__ == '__main__':

	rawData, rawLabels = extract_raw()
	trainingData, labels = raw_2_numpy(rawData, rawLabels)
	total = trainingData.shape[0]
	trainingData, labels = data_batches(trainingData, labels, BATCH_SIZE)

	layer1 = Dense(trainingData.shape[2],16)
	activation1 = Sigmoid()
	layer2 = Dense(16,10)
	activation2 = SoftMax()
	cost = CostMeanSquared()

for epoch in range(EPOCHS):
	print('Epoch: '+str(epoch+1)+'/'+str(EPOCHS))
	print('')
	correct = 0
	for batch in range(total//BATCH_SIZE):

		### SOCHASIC GRADIENT DESCENT ###

		layer1.forward(trainingData[batch])
		activation1.forward(layer1.outputs)
		layer2.forward(activation1.outputs)
		activation2.forward(layer2.outputs)
		cost.forward(activation2.outputs,labels[batch],10)

		for sample in range(activation2.outputs.shape[1]):
			if np.argmax(activation2.outputs[:,sample]) == np.argmax(labels[batch,sample]):
				correct +=1

		cost.backward(activation2.outputs,labels[batch],10)
		activation2.backward(layer2.outputs,layer2.weights.shape[0],BATCH_SIZE)
		layer2.backward(activation1.outputs)
		activation1.backward(layer1.outputs)
		layer1.backward(trainingData[batch])

		delta1 = np.zeros((cost.prime.shape[0],cost.prime.shape[1]))
		for i in range(cost.prime.shape[0]):
			delta1[i] = np.matmul(cost.prime[i], activation2.prime[i])

		delta1_wrt_L2 = np.matmul(delta1, layer2.input_prime)
		delta2 = np.zeros((activation1.prime.shape[0],activation1.prime.shape[2]))
		for i in range(activation1.prime.shape[2]):
			delta2[:,i] = np.matmul(delta1_wrt_L2[i],activation1.prime[:,:,i])

		C_wrt_W2 = np.zeros((delta1.shape[0],delta1.shape[1],layer2.weights_prime.shape[1]))
		for i in range(delta1.shape[0]):
			C_wrt_W2[i] = np.outer(delta1[i], layer2.weights_prime[i])
		C_wrt_W2 = np.sum(C_wrt_W2,axis=0)/BATCH_SIZE

		C_wrt_B2 = delta1
		C_wrt_B2 = np.sum(C_wrt_B2, axis = 0)/BATCH_SIZE
		C_wrt_B2 = np.array([C_wrt_B2])

		C_wrt_W1 = np.zeros((delta2.shape[1],delta2.shape[0],layer1.weights_prime.shape[1]))
		for i in range(delta2.shape[1]):
			C_wrt_W1[i] = np.outer(delta2[:,i], layer1.weights_prime[i])
		C_wrt_W1 = np.sum(C_wrt_W1,axis = 0)/BATCH_SIZE

		C_wrt_B1 = delta2
		C_wrt_B1 = np.sum(C_wrt_B1, axis = 1)/BATCH_SIZE
		C_wrt_B1 = np.array([C_wrt_B1])

		layer1.weights -= LEARNING_RATE*(C_wrt_W1)
		layer2.weights -= LEARNING_RATE*(C_wrt_W2)
		layer1.biases -= LEARNING_RATE*(C_wrt_B1.T)
		layer2.biases -= LEARNING_RATE*(C_wrt_B2.T)

	print('Loss: {:<20.9f}Accuracy: {:<20.9f}'.format(cost.outputs,correct/total))
	print('')
