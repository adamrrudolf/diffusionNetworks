
from dipy.tracking.eudx import EuDX
from dipy.reconst import peaks, shm
from dipy.tracking import utils

from dipy.data import read_stanford_labels, fetch_stanford_t1, read_stanford_t1

import numpy as np


 
#part for calculating streamlines from the diffusion data

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

# to include only longer streamlines, so we visualize lesser amount of tract, for hardware reasons

# part for changing including streamlines only longer than  particular length, here 50

from dipy.tracking.metrics import length  

longer_streamlines = []
for tract in streamlines:
    if length(tract)>50.0:
        longer_streamlines.append(tract)


# Streamlines visualization

from dipy.viz import fvtk
from dipy.viz.colormap import line_colors

# Make display objects

streamlines_actor = fvtk.line(longer_streamlines, line_colors(longer_streamlines))

# Add display objects to canvas
r = fvtk.ren()
fvtk.add(r, streamlines_actor)

# Save figure
fvtk.camera(r, [-1, 0, 0], [0, 0, 0], viewup=[0, 0, 1])
fvtk.record(r, n_frames=1, out_path='streamlines_saggital.png',size=(800, 800))
