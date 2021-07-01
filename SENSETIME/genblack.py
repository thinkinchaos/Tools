import os,sys
import h5py
import numpy as np
from scipy import io
import rawpy
from tqdm import tqdm
from matplotlib import pyplot as plt
imgpath = 'supernight_pipeline_fusion_out_bayer_4608_3456_1_float.raw'
raw = np.fromfile(imgpath, dtype='float32')
bayer_raw = raw.reshape(4608,3456)
print(bayer_raw[8-1][9-1])
bayer_raw = bayer_raw[:4, :4]
print(bayer_raw.shape, bayer_raw)
plt.imshow(bayer_raw)
plt.show()
