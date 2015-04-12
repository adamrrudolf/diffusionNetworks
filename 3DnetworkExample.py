
'''Example network visualisation using actors from Dipy fvtk models 


'''



import dipy.viz.fvtk as fvtk
import numpy as np



label_coords = np.loadtxt('labels_coords_86.txt')



labelsConnectivity = np.loadtxt('allconnectivityMatrix.txt')



lines_color = [205/255.0,247/255.0,255/255.0]
points_color = [2/255.0, 128/255.0, 232/255.0]

lines = []
for columnNumber in range(86):
    for rowNumber in range(86):
        if labelsConnectivity[columnNumber][rowNumber] > 20 :
            lines.append([label_coords[columnNumber],label_coords[rowNumber]])


ren = fvtk.ren()
pointActors = fvtk.point(label_coords, points_color, opacity=0.8, point_radius=3)
lineActors = fvtk.line(lines, lines_color, opacity=0.2, linewidth=2)


fvtk.add(ren, pointActors)

fvtk.add(ren, lineActors)


#to explore the data in 3D interactive way
#fvtk.show(ren)


#save figure

fvtk.camera(ren, [-1, -1, 0], [0, 0, 0], viewup=[0, 0, 1])
fvtk.record(ren, n_frames=1, 
            out_path='brain_network_example.png',
            size=(600, 600))
