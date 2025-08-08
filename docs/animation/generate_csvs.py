from flooder import generate_figure_eight_points_2d, flood_complex, generate_landmarks, generate_swiss_cheese_points
import matplotlib.pyplot as plt
import numpy as np

pts = generate_figure_eight_points_2d(200, centers=((0.3,.5), (0.6,0.5)))
# pts, _ ,_ = generate_swiss_cheese_points(250, (0,0),(1,1))
lms = generate_landmarks(pts, 25)
f_dict = flood_complex(pts, lms)
edges = [(*i, j) for i, j in f_dict.items() if len(i)==2]
triangles = [(*i, j) for i, j in f_dict.items() if len(i)==3]

np.savetxt('points.csv', pts, delimiter=',', fmt='%.8f')
np.savetxt('landmarks.csv', lms, delimiter=',', fmt='%.8f')
np.savetxt('edges.csv', edges, delimiter=',', fmt='%.8f')
np.savetxt('triangles.csv', triangles, delimiter=',', fmt='%.8f')