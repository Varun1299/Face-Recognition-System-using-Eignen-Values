import numpy as np
import cv2 as cv
import os

def dataFetcher():
	"""
	Traverses the datset directory.
	Converts each image to a 2D matrix of dimension N*N.
	Flattens the 2D matrix to 1D of dimension N^2 * 1.
	Returns a list of 1D matrices of all the training images. 
	"""
	imgVectorList = []
	dataDir = "orl_faces"
	fileList = os.listdir(dataDir)
	for item in fileList:
		itemPath = os.path.join(dataDir, item)
		if os.path.isdir(itemPath):
			subFileList = os.listdir(itemPath)
			for img in subFileList:
				imgPath = os.path.join(dataDir,item, img)
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
	Computes the eigrn values and eigen vectors of the covariance matrix
	"""
	altMatrix = np.dot(np.array(OffsetVectorList),np.array(OffsetVectorList).transpose())
	eigenValueArray, eigenVectorMatrix = np.linalg.eig(altMatrix)
	eigenVectorList = []
	for i in range(len(eigenVectorMatrix)):
		eigenVectorList.append(np.dot(np.array(OffsetVectorList).transpose(),eigenVectorMatrix[i]))
	return eigenValueArray.tolist(), eigenVectorList

def selectKeigenVectors(eigenValueList, eigenVectorList, K = 300):
	"Selects K eigen vectors corresponding to maximum K eigen values"
	selectedEigenVectors = []
	while len(selectedEigenVectors) < K:
		maxEigenValue = max(eigenValueList)
		index = eigenValueList.index(maxEigenValue)
		selectedEigenVectors.append(eigenVectorList[index])
		del eigenVectorList[index]
		del eigenValueList[index]
	return selectedEigenVectors

if __name__ == "__main__":
	imgVectorList = dataFetcher()
	print len(imgVectorList)
	avgImgVector = findAvgImgVector(imgVectorList)
	print avgImgVector.shape
	OffsetVectorList = makeOffsetVectorList(imgVectorList, avgImgVector)
	print len(OffsetVectorList), OffsetVectorList[0].shape
	covarianceMatrix = makeCovarianceMatrix(OffsetVectorList)
	print covarianceMatrix.shape
	eigenValueList, eigenVectorList = computeEigenVectors(OffsetVectorList)
	selectedEigenVectors = selectKeigenVectors(eigenValueList, eigenVectorList)
	print len(selectedEigenVectors)