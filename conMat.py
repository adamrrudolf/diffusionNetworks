from dipy.tracking import utils

import numpy as np

from nibabel import trackvis
from dipy.data import read_stanford_labels

hardi_img, gtab, labels_img = read_stanford_labels()
labels = labels_img.get_data()

affine = np.array([[ 1.,  0.,  0.,  0.],
       [ 0.,  1.,  0.,  0.],
       [ 0.,  0.,  1.,  0.],
       [ 0.,  0.,  0.,  1.]])

streams, hdr = trackvis.read('/home/arybinski/data/experimental_detr.trk')
streamlines = [s[0] for s in streams]



M, grouping = utils.connectivity_matrix(streamlines, labels, affine=affine,
                                        return_mapping=True,
                                        mapping_as_streamlines=True)
M[:3, :] = 0
M[:, :3] = 0


# Matrix including only 86 gray matter labels

labelsConnectivity = M[3:, 3:]

#make self-label connection equal 0
for i in range(86):
    labelsConnectivity[i][i] = 0

 
# Visualize matrix

import matplotlib.pyplot as plt


plt.imshow(np.log1p(M), interpolation='nearest')
plt.show()
#plt.savefig("allconnectivity.png")
#np.savetxt('allconnectivityMatrix.txt', labelsConnectivity)
