import numpy as np
import cv2 as cv
import sys
import os

def dataFetcher():
	"""
	Traverses the datset directory.
	Converts each image to a 2D matrix of dimension N*N.
	Flattens the 2D matrix to 1D of dimension N^2 * 1.
	Returns a list of 1D matrices of all the training images. 
	"""
	imgVectorList = []
	for i in range(1,41):
		for j in range(1,10):
			imgPath = os.path.join("orl_faces", "s" + str(i), str(j) + ".pgm")
			imgMatrix = cv.imread(imgPath, 0)
			imgVector = imgMatrix.flatten()
			imgVectorList.append(imgVector)
	return imgVectorList


def findAvgImgVector(imgVectorList):
	"""
	Computes and returns the mean of all the image vectors. 
	"""
	sumVector = imgVectorList[0].astype(np.uint32)
	for index in range(1,len(imgVectorList)):
		sumVector = sumVector.astype(np.uint32) + imgVectorList[index].astype(np.uint32)
	return sumVector/float(len(imgVectorList))

def makeOffsetVectorList(imgVectorList, avgImgVector):
	"""
	Subtracts avgImgVector from each image vector in imgVectorList
	and returns a list of the computed offset vectors. 
	"""
	OffsetVectorList = []
	for vector in imgVectorList:
		OffsetVectorList.append(vector - avgImgVector)
	return OffsetVectorList

def makeCovarianceMatrix(OffsetVectorList):
	"""
	Returns the covariance matrix
	"""
	return np.dot(np.array(OffsetVectorList).transpose(),np.array(OffsetVectorList))

def computeEigenVectors(OffsetVectorList):
	"""
	Computes the eigen values and eigen vectors of the covariance matrix
	"""
	altMatrix = np.dot(np.array(OffsetVectorList),np.array(OffsetVectorList).transpose())
	eigenValueArray, eigenVectorMatrix = np.linalg.eig(altMatrix)
	eigenVectorList = []
	for i in range(len(eigenVectorMatrix)):
		eigenVectorList.append(np.dot(np.array(OffsetVectorList).transpose(),eigenVectorMatrix[i]))
	for eigenVector in eigenVectorList:
		eigenVector = eigenVector/float(np.linalg.norm(eigenVector))
	return eigenValueArray.tolist(), eigenVectorList

def selectKeigenVectors(eigenValueList, eigenVectorList, K = 20):
	"""
	Selects K eigen vectors corresponding to maximum K eigen values
	"""
	selectedEigenVectors = []
	while len(selectedEigenVectors) < K:
		maxEigenValue = max(eigenValueList)
		index = eigenValueList.index(maxEigenValue)
		selectedEigenVectors.append(eigenVectorList[index])
		del eigenVectorList[index]
		del eigenValueList[index]
	return selectedEigenVectors

def makeWeightVectors(selectedEigenVectors, OffsetVectorList):
	"""
	Projects all the image vectors in space of the selected eigen vectors and returns the list of weight vectors.
	"""
	weightVectorList = []
	selectedEigenMatrix = np.array(selectedEigenVectors)
	for offsetVector in OffsetVectorList:
		weightVectorList.append(np.dot(selectedEigenMatrix,offsetVector))
	return weightVectorList


def train():
	print("Training...")
	sys.stdout.write('\r')
	sys.stdout.write("[%-70s] %d%%" % ('='*0, 0))
	imgVectorList = dataFetcher()
	sys.stdout.flush()
	sys.stdout.write('\r')
	sys.stdout.write("[%-70s] %d%%" % ('='*10, 1*(100/7)))
	avgImgVector = findAvgImgVector(imgVectorList)
	sys.stdout.flush()
	sys.stdout.write('\r')
	sys.stdout.write("[%-70s] %d%%" % ('='*20, 2*(100/7)))
	offsetVectorList = makeOffsetVectorList(imgVectorList, avgImgVector)
	sys.stdout.flush()
	sys.stdout.write('\r')
	sys.stdout.write("[%-70s] %d%%" % ('='*30, 3*(100/7)))
	covarianceMatrix = makeCovarianceMatrix(offsetVectorList)
	sys.stdout.flush()
	sys.stdout.write('\r')
	sys.stdout.write("[%-70s] %d%%" % ('='*40, 4*(100/7)))
	eigenValueList, eigenVectorList = computeEigenVectors(offsetVectorList)
	sys.stdout.flush()
	sys.stdout.write('\r')
	sys.stdout.write("[%-70s] %d%%" % ('='*50, 5*(100/7)))
	try:
		selectedEigenVectors = selectKeigenVectors(eigenValueList, eigenVectorList)
	except:
		selectedEigenVectors = selectKeigenVectors(eigenValueList, eigenVectorList)
	sys.stdout.flush()
	sys.stdout.write('\r')
	sys.stdout.write("[%-70s] %d%%" % ('='*60, 6*(100/7)))
	weightVectorList = makeWeightVectors(selectedEigenVectors, offsetVectorList)
	sys.stdout.flush()
	sys.stdout.write('\r')
	sys.stdout.write("[%-70s] %d%%" % ('='*70, 100))
	sys.stdout.write("\n")
	return avgImgVector, np.array(selectedEigenVectors), weightVectorList


def recognizeFace(avgImgVector, selectedEigenMatrix, weightVectorList, testImgPath):
	"""
	Computes disatance of test image with each image in dataset and accordingly outputs the predicted face.
	"""
	thresholdValue = 25000000
	testImg = cv.imread(testImgPath,0).flatten()
	normalizedTestImg = testImg - avgImgVector
	weightedTestImg = np.dot(selectedEigenMatrix,normalizedTestImg)
	minDist = np.inf
	minDistIndex = -1
	for index in range(len(weightVectorList)):
		if np.linalg.norm(weightedTestImg - weightVectorList[index]) < minDist:
			if np.linalg.norm(weightedTestImg - weightVectorList[index]) < thresholdValue:
				minDist = np.linalg.norm(weightedTestImg - weightVectorList[index])
				minDistIndex = index
	if minDist > thresholdValue:
		print("Could not recognize!")
	return "s" + str(int(minDistIndex/9) + 1)	

if __name__ == "__main__":
	avgImgVector, selectedEigenMatrix,  weightVectorList = train()
	try:
		testImgPath = sys.argv[1]
		print("Recognition started...")
		print( recognizeFace(avgImgVector, selectedEigenMatrix, weightVectorList, testImgPath))
	except:
		for i in range(1,41):
			testImgPath = os.path.join("orl_faces", "s" + str(i), "10.pgm")
			predictedFace = recognizeFace(avgImgVector, selectedEigenMatrix, weightVectorList, testImgPath)
			print("s" + str(i), predictedFace, "s" + str(i) == predictedFace)