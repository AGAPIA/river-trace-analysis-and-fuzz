
import argparse
from ModelDef import *
from enum import Enum
import os
import Utils_globals
import numpy as np
from testDataType import TestDataType
import shutil
import Utils_regularExpr

class InferenceType(Enum):
    FIXED_ORDER = 1,
    FIXED_OFFSETS = 2


def inference(modelFolderPath, inferenceType : InferenceType, maxPercentToExtend : float, numTests : int, limitForStarGrammarGeneration : int):
    print ("DEBUG inference: model {} inferenceType {} percenTo extend {} numTests {} limit start gen characters {}".format(modelFolderPath, inferenceType, maxPercentToExtend, numTests, limitForStarGrammarGeneration))
    assert (maxPercentToExtend >= 1.0), "maxPercentToExtend must be >= 1.0 !!"

    fullModelPath = os.path.join(modelFolderPath, Utils_globals.SINGLE_MODEL_FILENAME)
    folderWithResults = os.path.join(modelFolderPath, "results")
    if os.path.exists(folderWithResults):
        shutil.rmtree(folderWithResults)
    os.mkdir(folderWithResults)


    model = ParsingResult.loadModel(modelFolderPath)
    listOfExampleTests  = model.listOfTestsExtractedData
    numExampleTests= len(listOfExampleTests)

    # Generate tests one by one
    for testIter in range(0, numTests):

        # Choose one saved in the model as example
        testExample : TestDataType = listOfExampleTests[np.random.randint(0, numExampleTests)]

        # Build a random input string first
        testLength = testExample.length if inferenceType == InferenceType.FIXED_OFFSETS else int(testExample.length * maxPercentToExtend)
        r = np.random.randint(0, 256, testLength)
        strResult = ''.join(chr(i) for i in r)

        numBlocksInTestExample = len(testExample.blockOffsets)

        # If fixed offsets is preferred then use the same offsets for generating blocks of data
        if inferenceType == InferenceType.FIXED_OFFSETS:
            # Generate all blocks with given offsets
            for blockIter in range(0, numBlocksInTestExample):
                blockIndex = testExample.blockIndicesUsed[blockIter]
                blockOffset = testExample.blockOffsets[blockIter]
                blockGrammar = model.blockIndexToGrammarString[blockIndex]

                blockResult = Utils_regularExpr.infereRegularExpression(blockGrammar, limit = limitForStarGrammarGeneration)

                # Replace the string place
                blockEndInString = min(blockOffset + len(blockResult), len(strResult))
                strResult = strResult[:blockOffset] + blockResult +  strResult[blockEndInString:]


        else: # Generate sequences in order and put them randomly
            assert (False), "todo"

        testFullPath = os.path.join(folderWithResults, str(testIter)+".txt")
        with open(testFullPath, "w") as f:
            f.write(strResult)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="inference script. Give it the folder path to data and if fixed order or offsets")
    parser.add_argument('-path', action='store', default="./data/model_httpparser", help="path to the training data folder", dest="folderPath")
    parser.add_argument('-inferenceType', action='store', default="FIXED_OFFSETS",
                        help="FIXED_ORDER or FIXED_OFFSETS. If you set FIXED_ORDER you have to give the length max percent param too", dest="inferenceType")

    parser.add_argument('-maxLenPercentToExtend', action='store', default="100", dest="maxLenPercentToExtend",
                        help="the maximum length over the original test that can be generated")

    parser.add_argument('-n', action='store', default=10, type=int, help="number of tests to generate", dest="numTests")

    parser.add_argument('-starLimit', action='store', default=7, type=int, help="the limit on number of characters to generate when + or * symbol is used", dest="limitForStarGrammarGeneration")

    params = parser.parse_args()

    #print(results.folderPath)

    inference(params.folderPath, InferenceType[params.inferenceType], float(params.maxLenPercentToExtend)/100, params.numTests, params.limitForStarGrammarGeneration)

