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

cdef calculate_pixel_gradient_magnitude(np.ndarray image, int y, int x):
    cdef double gx, gy, gz, gm

    if image.ndim == 2:
        if y + 1 < image.shape[0]:
            gy = image[y, x] - image[y + 1, x]
        else:
            gy = image[y, x]

        if x + 1 < image.shape[1]:
            gx = image[y, x] - image[y, x + 1]
        else:
            gy = image[y, x]

        gm = ((gx * gx) + (gy * gy)) ** 0.5

    return gm

cdef calculate_pixel_mean_curvature(np.ndarray image, int x, int y):
    cdef double fx, fy, fxx, fyy, fxy, curvature
    cdef int h, k
    h = 1
    k = 1
    if image.ndim == 2:
        fx = (image[y, x + h] - image[y, x - h]) / (2*h)

        fy = (image[y + k, x] - image[y - k, x]) / (2*k)

        fxx = (image[y, x + h] - 2*image[y,x] + image[y, x - h]) / (h*h)

        fyy = (image[y + k, x] - 2*image[y,x] + image[y - k, x]) / (k*k)

        fxy = (image[y + k, x + h] - image[y - k, x + h] - image[y + k, y - h] + image[y - k, x - h]) / (4*h*k)

        curvature = (fxx*fy*fy - 2*fxy*fx*fy + fyy*fx*fx) / (fx*fx + fy*fy)**1.5

    return curvature

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
                        out[i, j] = calculate_pixel_gradient_magnitude(out, i, j)
    else:
        pass

    return out
