# Currently we expect to have a folder InputAndLogs in the script's folder
# with the following pairs of files (input0, log0), (input1, log1), ....
# Inputs are in inputs subfolder, while logs in the logs one
# They must have the same name. e.g. inputs/1,  logs/1 | inputs/test7, logs/test7
# Logs are obtained after running the sh script with inputs (as text) from a database


from fileblockparser import FileBlockParser
from testDataType import *
import os
import Utils_regularExpr
from ModelDef import *
import argparse

def unittests(parser : FileBlockParser):
	intervals = [0, 32, 64, 128, 512, 1024, 1400, 1700, 1800, 2000]

	print (parser.cutSubInterval(intervals, [100,1750]))

	intervals = [0, 32]

	print (parser.cutSubInterval(intervals, [16,32]))


# returns a list of TestDataTypes with extracted information from each test, the forward and backward mapping from block offsets strings to ids
def ParseFolder(folderToParse):
    inputsFolder = os.path.join(folderToParse, "inputs")
    logsFolder = os.path.join(folderToParse,"logs")

    # A dictionary with items of type (blockOffset : string, blockId : int)
    dictBlockToIndex = { }
    globalBlockIndex = 0

    listOfTestsExtractedData : list[TestDataType] = []
    for fileName in os.listdir(inputsFolder):
        input_fullPath = os.path.join(inputsFolder, fileName)
        log_fullPath = os.path.join(logsFolder, fileName)

        if not os.path.isfile(input_fullPath):
            continue

        assert (os.path.isfile(log_fullPath)), "Filename {} was found in inputs but not in logs !".format(fileName)

        parser = FileBlockParser()
        blocksPerLog = parser.run(log_fullPath)
        #print(blocksPerLog)

        newTestEntry = TestDataType()

        for blockOffset, blockData in blocksPerLog.items():
            blockIndex = -1
            if blockOffset not in dictBlockToIndex:
                globalBlockIndex += 1
                dictBlockToIndex[blockOffset] = globalBlockIndex

            blockIndex = globalBlockIndex

            intervalsUsed = blockData[0][1]
            newTestEntry.addNewBlockData(blockIndex, intervalsUsed)

        newTestEntry.loadTempData(input_fullPath)
        listOfTestsExtractedData.append(newTestEntry)

    dictIndexToBlock = {blockIndex:blockOffset for blockOffset,blockIndex in dictBlockToIndex.items()}

    result = ParsingResult()
    result.dictIndexToBlock = dictIndexToBlock
    result.dictBlockToIndex = dictBlockToIndex
    result.numBlocks = globalBlockIndex
    result.listOfTestsExtractedData = listOfTestsExtractedData

    return result

def findGrammarsPerBlock(parseResult : ParsingResult):

    # For each block index gather the list of strings used and find a grammar matching all
    for blockIndex in range(1, parseResult.numBlocks + 1):
        listOfStringsUsed = [] # Gather the list of all strings used for this block index in test files

        # For each test find the string used for this blockIndex and add it to list (if any)
        for testEntry  in parseResult.listOfTestsExtractedData:
            intervalsUsedByBlockIndex = testEntry.blockIndexToIntervalsUsed.get(blockIndex)
            if intervalsUsedByBlockIndex == None:
                continue

            stringUsedInTest = ""
            for intervalIndex in range(0, len(intervalsUsedByBlockIndex), 2):
                beginInt = intervalsUsedByBlockIndex[intervalIndex]
                endInt = intervalsUsedByBlockIndex[intervalIndex + 1]
                stringUsedInTest += testEntry.tempInputStr[beginInt:endInt+1]


            listOfStringsUsed.append(stringUsedInTest)

        assert (listOfStringsUsed is not None), "There is no one using this blockIndex ??? It should be used somewhere because otherwise it won't appear here"
        # Fit the blockIndex grammar
        blockGrammar = Utils_regularExpr.getRegularExpressionForStrings(listOfStringsUsed)
        parseResult.blockIndexToGrammarString[blockIndex] = blockGrammar


def buildModel(baseFolder):
    # Parse the inputs/logs folder and find a list with all inputs structure extracted from logs, and a dictionary mapping offsets in files to binary
    parseResult = ParseFolder(os.path.join(baseFolder, "InputsAndLogs"))

    # Find grammar per block
    findGrammarsPerBlock(parseResult)

    # Cleanup test data
    for testEntry in parseResult.listOfTestsExtractedData:
        testEntry.unloadTempData()

    parseResult.saveModel(baseFolder)


    # Test things
    parseResult2 = ParsingResult.loadModel(baseFolder)
    #testEqual = parseResult == parseResult2
    #print("Save / load succeded ? {}".format(testEqual))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="training script. Give it the folder path to data")
    parser.add_argument('-path', action='store', default="./data/model_httpparser", help="path to the training data folder", dest="folderPath")
    results = parser.parse_args()

    #print(results.folderPath)

    buildModel(results.folderPath)

    #unittests(parser)

