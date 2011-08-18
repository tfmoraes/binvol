import mahotas
import numpy as np
cimport numpy as np

N = 100
THRESHOLD = 0.0001

def perim(np.ndarray image):
    if image.ndim == 2:
        # Connectivity 8
        return mahotas.labeled.borders(image, np.ones((3, 3)))
    elif image.ndim == 3:
        # Connectivity 26
        return mahotas.labeled.borders(image, np.ones((3, 3, 3)))

def smooth(np.ndarray image, int n):
    cdef double dt, gx, gy
    cdef int i, j, k, p
    cdef np.ndarray A0, A1, A2, A3, A
    cdef np.ndarray out

    if not image.ndim in (2, 3):
        raise ValueError("Only 2 and 3 dimensions are supported")

    A0 = perim(image)
    A1 = perim(A0)
    A2 = perim(A1)
    A3 = perim(A2)
    A = A0 | A1 | A2 | A3

    out = image.astype('double')

    if image.ndim == 2:
        for p in xrange(n):
            for i in xrange(image.shape[0]):
                for j in xrange(image.shape[1]):
                    if A[i, j]:
                        if i + 1 < image.shape[0]:
                            gx = out[i,j] - out[i+1, j]
                        else:
                            gx = image[i, j]

                        if j + 1 < image.shape[1]:
                            gy = out[i,j] - out[i, j + 1]
                        else:
                            gy = out[i, j]

                        out[i, j] = ((gx * gx) + (gy * gy)) ** 0.5
    else:
        pass

    return out
