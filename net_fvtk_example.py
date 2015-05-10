
import dipy.viz.fvtk as fvtk
import numpy as np


pointA = [0.48412286, 0.97713726, 0.74590314]
pointB = [0.07821987, 0.13495185, 0.7239567]
pointC = [0.79723847, 0.06467979, 0.45524464]
pointD = [0.25058964, 0.83944661, 0.16528851]
pointE = [0.61336066, 0.28185135, 0.94522671]


example_xyz = np.array([pointA, 
                        pointB,
                        pointC,
                        pointD,
                        pointE])


ren = fvtk.ren()
point_actor = fvtk.point(example_xyz, fvtk.colors.blue_light)
fvtk.add(ren, point_actor)

lineAB = np.array([pointA, pointB])
lineBC = np.array([pointB, pointC])
lineCD = np.array([pointC, pointD])
lineCE = np.array([pointC, pointE])

line_color = [0.9, 0.97, 1.0]
line_actor_AB = fvtk.line(lineAB, line_color)
line_actor_BC = fvtk.line(lineBC, line_color)
line_actor_CD = fvtk.line(lineCD, line_color)
line_actor_CE = fvtk.line(lineCE, line_color)


fvtk.add(ren, line_actor_AB)
fvtk.add(ren, line_actor_BC)
fvtk.add(ren, line_actor_CD)
fvtk.add(ren, line_actor_CE)


#fvtk.show(ren)

fvtk.camera(ren, [-1, -1, 0], [0, 0, 0], viewup=[0, 0, 1])
fvtk.record(ren, n_frames=1, 
            out_path='simple_network_example.png',
            size=(400, 400))
