
from dipy.tracking.eudx import EuDX
from dipy.reconst import peaks, shm
from dipy.tracking import utils

import numpy as np

from dipy.data import read_stanford_labels, fetch_stanford_t1, read_stanford_t1

hardi_img, gtab, labels_img = read_stanford_labels()
data = hardi_img.get_data()
labels = labels_img.get_data()



fetch_stanford_t1()
t1 = read_stanford_t1()
t1_data = t1.get_data()

white_matter = (labels == 1) | (labels == 2)
csamodel = shm.CsaOdfModel(gtab, 6)
csapeaks = peaks.peaks_from_model(model=csamodel,
                                  data=data,
                                  sphere=peaks.default_sphere,
                                  relative_peak_threshold=.8,
                                  min_separation_angle=45,
                                  mask=white_matter)

seeds = utils.seeds_from_mask(white_matter, density=2)
streamline_generator = EuDX(csapeaks.peak_values, csapeaks.peak_indices,
                            odf_vertices=peaks.default_sphere.vertices,
                            a_low=.05, step_sz=.5, seeds=seeds)
affine = streamline_generator.affine
streamlines = list(streamline_generator)


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
#plt.show()
plt.savefig("allconnectivity.png")
np.savetxt('allconnectivityMatrix.txt', labelsConnectivity)
