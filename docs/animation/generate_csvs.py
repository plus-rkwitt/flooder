from flooder import generate_figure_eight_2d_points, flood_complex, generate_landmarks
import matplotlib.pyplot as plt
import numpy as np

pts = generate_figure_eight_2d_points(250)
lms = generate_landmarks(pts, 50)
f_dict = flood_complex(pts, lms)
edges = [(*i, j) for i, j in f_dict.items() if len(i)==2]
triangles = [(*i, j) for i, j in f_dict.items() if len(i)==3]

np.savetxt('points.csv', pts, delimiter=',', fmt='%.8f')
np.savetxt('landmarks.csv', lms, delimiter=',', fmt='%.8f')
np.savetxt('edges.csv', edges, delimiter=',', fmt='%.8f')
np.savetxt('triangles.csv', triangles, delimiter=',', fmt='%.8f')