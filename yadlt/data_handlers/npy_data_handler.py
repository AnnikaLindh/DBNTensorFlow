from yadlt.core.data_handler import DataHandler
import numpy as np
from numpy.lib.format import open_memmap

__author__ = 'Annika Lindh'


class NpyDataHandler(DataHandler):
    """
    Data handler that uses numpy npy files, with the option of using their memmapped version.
    """

    def __init__(self, dataFile, memmapped=True, copyBatch=False):
        self.memmapped = memmapped
        self.copyBatch = copyBatch
        self._loadData(dataFile)
        self.encodedData = None

    def _loadData(self, dataFile):
        self.data = np.load(dataFile, mmap_mode=('r' if self.memmapped else None))
        self.dataLength = len(self.data)
        self.indices = None
        self.nextIndex = 0

    def startEpoch(self, shuffle=True):
        if shuffle:
            self.indices = np.random.permutation(self.dataLength)
        else:
            self.indices = np.arange(0, self.dataLength)

        self.nextIndex = 0

    def nextBatch(self, batchSize):
        if self.indices is None:
            return None

        currentIndex = self.nextIndex
        self.nextIndex = min(self.nextIndex + batchSize, self.dataLength)
        batchIndices = self.indices[currentIndex:self.nextIndex]
        if self.nextIndex == self.dataLength:
            self.indices = None

        if self.copyBatch:
            return self.data[batchIndices, ].copy()
        else:
            return self.data[batchIndices, ]

    def startStoring(self, filepath, encodedSize, dtype, numRows=-1):
        if numRows < 0:
            numRows = self.dataLength
        self.encodedData = open_memmap(filepath, 'w+', dtype=dtype, shape=(numRows, encodedSize,))
        self.startEpoch(shuffle=False)

    def finishStoring(self, nextDataFile=None):
        del self.encodedData
        self.encodedData = None
        if nextDataFile is not None:
            self._loadData(nextDataFile)

    def store(self, data):
        self.encodedData[self.nextIndex-len(data):self.nextIndex, ] = data[:]

