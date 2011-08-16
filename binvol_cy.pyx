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
    cdef double dt
    cdef np.ndarray A0, A1, A2, A3, A

    if not image.ndim in (2, 3):
        raise ValueError("Only 2 and 3 dimensions are supported")

    A0 = perim(image)
    A1 = perim(A0)
    A2 = perim(A1)
    A3 = perim(A2)
    A = A0 | A1 | A2 | A3

    if image.ndim == 2:
        pass
    else image.ndim:
        pass

    return A
