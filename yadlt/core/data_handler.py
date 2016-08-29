__author__ = 'Annika Lindh'

class DataHandler(object):

    def startEpoch(self):
        pass

    def nextBatch(self, batchSize):
        pass

    def startStoring(self, filepath, encodedSize, dtype):
        pass

    def finishStoring(self, nextDataFile=None):
        pass

    def store(self, data):
        pass
