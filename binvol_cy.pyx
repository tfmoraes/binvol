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
            gx = image[y, x]

        gm = ((gx * gx) + (gy * gy)) ** 0.5

    return gm

cdef calculate_pixel_mean_curvature(np.ndarray image, int x, int y):
    cdef double fx, fy, fxx, fyy, fxy, curvature
    cdef int h, k
    h = 1
    k = 1
    if image.ndim == 2:
        fx = (image[y, x + h] - image[y, x - h]) / (2.0*h)

        fy = (image[y + k, x] - image[y - k, x]) / (2.0*k)

        fxx = (image[y, x + h] - 2*image[y,x] + image[y, x - h]) / (h*h)

        fyy = (image[y + k, x] - 2*image[y,x] + image[y - k, x]) / (k*k)

        fxy = (image[y + k, x + h] - image[y - k, x + h] - image[y + k, y - h] + image[y - k, x - h]) / (4.0*h*k)

        curvature = (fxx*(1 + fy*fy) - 2.0*fxy*fx*fy + fyy*(1+fx*fx)) /(2 * (1 + fx*fx + fy*fy)**1.5)

    return curvature

def smooth(np.ndarray image, int n):
    cdef double dt, gm, K, H
    cdef int i, j, k, p
    cdef np.ndarray A0, A1, A2, A3, A
    cdef np.ndarray out
    cdef np.ndarray tmp

    dt = 1/6.0

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
            tmp = out.copy()
            for i in xrange(image.shape[0]):
                for j in xrange(image.shape[1]):
                    if A[i, j]:
                        gm = calculate_pixel_gradient_magnitude(tmp, i, j)
                        K = calculate_pixel_mean_curvature(tmp, i, j)
                        H = gm * K
                        print H
                        if image[i, j]:
                            out[i, j] = max((tmp[i,j] + dt*H, 0.5))
                        else:
                            out[i, j] = min((tmp[i,j] + dt*H, 0.5))

    else:
        pass

    return out
