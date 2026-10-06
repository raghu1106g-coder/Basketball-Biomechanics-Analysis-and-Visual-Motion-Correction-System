import numpy as np
from fastdtw import fastdtw

A = np.array([160, 140, 120, 150])
B = np.array([160, 150, 140, 120, 150])

distance, path = fastdtw(A, B, dist=lambda x, y: abs(x - y))
print("DTW distance:", distance)      # should print 10
print("Alignment path:", path)        # should show A[2]<->B[3] (0-indexed)