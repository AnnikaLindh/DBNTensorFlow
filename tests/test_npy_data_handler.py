import sys
from subprocess import call
import numpy as np
from PIL import Image
from yadlt.data_handlers.npy_data_handler import NpyDataHandler

__author__ = 'Annika Lindh'


try:
    dataFile = sys.argv[1]
    batchSize = int(sys.argv[2])
    outDir = sys.argv[3]
    targetSize_side = int(sys.argv[4])
except:
    print "Usage: python test_npy_data_handler <dataset-file.npy> <batch-size> <out-dir> <target-size-side>"
    raise

data = NpyDataHandler(dataFile, True, True)
data.startEpoch()
iBatch = 0
batch = data.nextBatch(batchSize)
while batch is not None:
    batchDir = outDir + '/batch_' + str(iBatch)
    call(["mkdir", "-p", batchDir])
    iRow = 0
    for row in batch:
        for i in range(0, 5):
            start = i*targetSize_side*targetSize_side
            imgArr = (row[start:start+targetSize_side*targetSize_side])
            imgArr = np.array(imgArr*255, dtype=np.uint8)
            img = Image.frombuffer('L', [targetSize_side,targetSize_side], imgArr.tostring(), 'raw', 'L', 0, 1)
            img.save(batchDir + "/img_" + str(iRow) + "_" + str(i) + ".png")
        iRow += 1
    iBatch += 1
    batch = data.nextBatch(batchSize)
