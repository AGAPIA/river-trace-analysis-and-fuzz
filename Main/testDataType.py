# This class stores data about each individual test
# For instance if input is [a0a1......aN] then we store the block indices used when running against the input, the offset (start) of each block, the len of the input

import math
import os

class TestDataType:
    def __init__(self):
        self.length = 0 # total length of input in bytes

        # Those are AoS defining metadata on each block appearing on this input
        self.blockIndicesUsed = [] # List of all blocks used in order of their appearance
        self.blockOffsets = [] # The offset in bytes (from beginning ) of appearance of the block
        self.blockIndexToIntervalsUsed = {} #  x[blockIndex] = y => blockIndex uses intervals specified in list y


        self.tempInputStr = "" # This is used during model training time to load the initial input that generated the log for this test. It is deleted after training, and not available at inference

    def processIntervals(self, intervals):
        numIntervals = len(intervals)
        for i in range(0, numIntervals, 2):
            beginIn = math.floor(float(intervals[i]) / 8)
            endIn = math.ceil(float(intervals[i+1]) / 8)-1
            endIn = max(endIn, beginIn)

            intervals[i] = beginIn
            intervals[i+1] = endIn


    def addNewBlockData(self, blockIndex, intervals):
        self.blockIndicesUsed.append(blockIndex)

        self.processIntervals(intervals)
        self.blockIndexToIntervalsUsed[blockIndex] = intervals
        self.blockOffsets.append(intervals[0])

    def loadTempData(self, inputFile):
        with open(inputFile, "r") as f:
            self.tempInputStr = f.read().replace('\n', ' ')

        self.length = len(self.tempInputStr)

    def unloadTempData(self):
        del self.tempInputStr


