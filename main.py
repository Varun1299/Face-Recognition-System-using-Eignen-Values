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

if __name__ == "__main__":
	imgVectorMatrix = dataFetcher()
	print imgVectorMatrix