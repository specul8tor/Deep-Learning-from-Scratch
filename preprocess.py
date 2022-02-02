import csv
import numpy as np

def extract_raw():
	rawData=[]
	rawLabels = []
	dataNumber = 0
	with open('train.csv','r') as csvfile:
		reader = csv.reader(csvfile)
		next(reader)
		for line in reader:
			rawData.append(line[1:])
			for pixel in range(len(line[1:])):
				rawData[dataNumber][pixel] = int(rawData[dataNumber][pixel])
			rawLabels.append(int(line[0]))
			dataNumber += 1
	return rawData, rawLabels

def raw_2_numpy(rawData,rawLabels):
	trainingData = np.array(rawData)
	trainingData = trainingData / 255
	labels = []
	for label in rawLabels:
		currentLabel = np.zeros(10)
		currentLabel[label] = 1
		labels.append(currentLabel)
	labels = np.array(labels)
	return trainingData, labels

def data_batches(trainingData, labels, batch_size):
	trainingData = np.reshape(trainingData,(trainingData.shape[0]//batch_size,batch_size,trainingData.shape[1]), order = 'F')
	labels = np.reshape(labels,(labels.shape[0]//batch_size,batch_size,labels.shape[1]))
	return trainingData, labels
