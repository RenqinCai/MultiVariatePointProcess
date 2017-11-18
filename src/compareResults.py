import numpy as np

def readEstimatedParam(paramFile):
	f = open(paramFile)

	paramList = []
	for rawLine in f:
		line = rawLine.strip().split("\t")

		lineLen = len(line)
		for lineIndex in range(lineLen):
			paramList.append(line[lineIndex])

	return paramList

def readTrueParam(lambdaFile, alphaFile):
	lambdaF = open(lambdaFile)

	alphaF = open(alphaFile)

	paramList = []

	for rawLine in lambdaF:
		line = rawLine.strip().split("\t")
		lineLen = len(line)

		for lineIndex in range(lineLen):
			paramList.append(line[lineIndex])

	lambdaF.close()

	for rawLine in alphaF:
		line = rawLine.strip().split("\t")
		lineLen = len(line)

		for lineIndex in range(lineLen):
			paramList.append(line[lineIndex])

	alphaF.close()

	return paramList

def computeRelError(trueParamList, estimatedParamList):
	paramNum = len(trueParamList)

	avgError = 0
	for paramIndex in range(paramNum):
		trueParam = trueParamList[paramIndex]
		estimatedParam = estimatedParamList[paramIndex]

		relErr = np.abs(trueParam-estimatedParam)
		print("relErr\t", relErr)

		if trueParam != 0:
			relErr = relErr/np.abs(trueParam)

		print("trueParam\t", trueParam)
		print("estimatedParam\t", estimatedParam)

		print("changing relErr\t", relErr)
		avgError += relErr
		print("++++++")
	
	print("avgError\t", avgError)
	avgError /= (paramNum*1.0)

	return avgError


alphaFile = "alpha.txt"
lambdaFile = "lambda.txt"

estimatedParamFile = "sparse_hawkes_estimatedParam.txt"

trueParamList = readTrueParam(lambdaFile, alphaFile)

estimatedParamList = readEstimatedParam(estimatedParamFile)

computeRelError(trueParamList, estimatedParamList)