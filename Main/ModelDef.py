import Utils_globals
import testDataType
import pickle
import os
9
# This contain the results after parsing an entire folder with tests and logs
class ParsingResult:
    def __init__(self):
        self.listOfTestsExtractedData = None  # List of TestDataType
        self.numBlocks = None  # Total numb blocks in all tests
        self.dictBlockToIndex = None  # dictionaries mapping block offsets to index and inverse
        self.dictIndexToBlock = None
        self.blockIndexToGrammarString = {}  # Mapping from block index to grammar string

    def saveModel(self, baseFolder):
        modelsFolderFullPath = os.path.join(baseFolder, "model")
        if not os.path.exists(modelsFolderFullPath):
            os.makedirs(modelsFolderFullPath)

        '''
        # Save the model files from parse result

        # Step 1 - grammar file
        with os.path.open(os.path.join(modelsFolderFullPath, Utils_globals.GRAMMAR_FILENAME), "w") as fgrammar:
            fgrammar.write(self.numBlocks)

            for blockIndex in range(1, self.numBlocks + 1):
                blockGrammar = self.blockIndexToGrammarString[blockIndex]
                fgrammar.write(blockGrammar)

        # Step 2 - write the tests dumps
        with os.path.open(os.path.join(modelsFolderFullPath, Utils_globals.TESTSDUMP_FILE_NAME), "w") as fTests:
            pickle.dump(self.listOfTestsExtractedData, fTests)

        # Step 3 - write the offsets corresponding to each block index
        with os.path.open(os.path.join(modelsFolderFullPath, Utils_globals.blocksOffsets), "w") as foffsets:
            foffsets.write(self.numBlocks)

            for blockIndex in range(1, self.numBlocks + 1):
                blockOffset = self.dictIndexToBlock[blockIndex]
                foffsets.write(blockOffset)
                
        '''
        with open(os.path.join(modelsFolderFullPath, Utils_globals.SINGLE_MODEL_FILENAME), "wb") as f:
            pickle.dump(self, f)

    def loadModel(baseFolder):

        modelsFolderFullPath = os.path.join(baseFolder, "model")
        if not os.path.exists(modelsFolderFullPath):
            print("Error: folder {} with models couldn't be found".format(modelsFolderFullPath))

        '''
        with os.path.open(os.path.join(modelsFolderFullPath, Utils_globals.GRAMMAR_FILENAME), "r") as fgrammar:
            self.numBlocks = int(fgrammar.readline())

            for blockIndex in range(1, self.numBlocks + 1):
                self.blockIndexToGrammarString = {}
                blockGrammar = fgrammar.readline()
                self.blockIndexToGrammarString[blockIndex] = blockGrammar

        # Step 2 - write the tests dumps
        with os.path.open(os.path.join(modelsFolderFullPath, Utils_globals.TESTSDUMP_FILE_NAME), "r") as fTests:
            pickle.dump(self.listOfTestsExtractedData, fTests)

        # Step 3 - write the offsets corresponding to each block index
        with os.path.open(os.path.join(modelsFolderFullPath, Utils_globals.blocksOffsets), "r") as foffsets:
            numBlocks = int(fgrammar.readline())
            assert (numBlocks == self.numBlocks), "in the grammar there were a different number of blocks"

            self.dictIndexToBlock = {}
            self.dictBlockToIndex = {}


            for blockIndex in range(1, numBlocks + 1):
                blockOffset = foffsets.readline()
                self.dictIndexToBlock[blockIndex] = blockOffset
                self.dictBlockToIndex[blockOffset] = blockIndex

        '''

        with open(os.path.join(modelsFolderFullPath, Utils_globals.SINGLE_MODEL_FILENAME), "rb") as f:
            data = pickle.load(f)
            return data

        return null

