import mahotas
import numpy as np
from scipy import ndimage

N = 100
THRESHOLD = 0.0001

def perim(image):
    if image.ndim == 2:
        # Connectivity 8
        return mahotas.labeled.borders(image, np.ones((3, 3)))
    elif image.ndim == 3:
        # Connectivity 26
        return mahotas.labeled.borders(image, np.ones((3, 3, 3)))

def smooth(image, n):
    dt = 1/6.0
    u = image.astype('float64')
    A0 = perim(image)
    A1 = perim(A0)
    A2 = perim(A1)
    A3 = perim(A2)
    A = A0 | A1 | A2 | A3
    n = 0

    if image.ndim == 2:
        while 1:
            gy, gx = np.gradient(u)

            # gradient magnitude
            g_mag = np.sqrt(gx**2 + gy**2)

            n_gy = gy / (g_mag + (g_mag == 0))
            n_gx = gx / (g_mag + (g_mag == 0))

            dy_n_gx, dx_n_gx = np.gradient(n_gx)
            dy_n_gy, dx_n_gy = np.gradient(n_gy)
            
            # mean curvature
            k = dx_n_gx + dy_n_gy

            # H is the gradient magnitude times the mean curvature
            H = g_mag * k

            nu = u.copy()
            temp_u = u + dt * H
            nu[(image) & (A)] = np.fmax(temp_u[(image) & (A)], 0.5)
            nu[(~image) & (A)] = np.fmin(temp_u[(~image) & (A)], 0.5)

            avg_change = (1.0/np.sum(A) * np.sum((nu[A] - u[A])**2)) ** 0.5
            print "avg", avg_change
            #print n, "BinVol", np.sum(gx**2 + gy**2)
            if avg_change <= THRESHOLD or n == N:
                break
            n += 1

            u = nu.copy()

            del H
            del temp_u

    elif image.ndim == 3:
        while 1:
            gz, gy, gx = np.gradient(u)

            # gradient magnitude
            g_mag = np.sqrt(gx**2 + gy**2 + gz**2)

            n_gz = gz / (g_mag + (g_mag == 0))
            n_gy = gy / (g_mag + (g_mag == 0))
            n_gx = gx / (g_mag + (g_mag == 0))

            dz_n_gz, dy_n_gz, dx_n_gz = np.gradient(n_gz)
            dz_n_gy, dy_n_gy, dx_n_gy = np.gradient(n_gy)
            dz_n_gx, dy_n_gx, dx_n_gx = np.gradient(n_gx)

            # mean curvature
            k = dz_n_gz + dx_n_gx + dy_n_gy

            # H is the gradient magnitude times the mean curvature
            H = g_mag * k

            nu = u.copy()
            temp_u = u + dt * H
            nu[(image) & (A)] = np.fmax(temp_u[(image) & (A)], 0.5)
            nu[(~image) & (A)] = np.fmin(temp_u[(~image) & (A)], 0.5)

            avg_change = (1.0/np.sum(A) * np.sum((nu[A] - u[A])**2)) ** 0.5
            print "avg", avg_change
            #print n, "BinVol", np.sum(gx**2 + gy**2)
            if avg_change <= THRESHOLD or n == N:
                break

            n += 1
            u = nu.copy()
            del H
            del temp_u
    else:
        raise NotImplemented
    return nu


def binvol(image):
    # image must be binary
    B = perim(image)
    return B

def __make_thorus():
    x,y,z = ogrid[0:75,0:75,0:75]
    c = 37 
    r2 = 40
    teta = pi/8.0
    R_t = 25
    r_t = 10
    thorus = (R_t - sqrt((x-c)**2+(y-c)**2))**2 + (z-c)**2 <= r_t**2
    return thorus

def __test_thorus():
    pass

def __test_lena():
    print "> Testing lena"
    from scipy.misc.common import lena
    from scipy.misc import imsave
    l = lena().astype('uint8')
    o = mahotas.otsu(l)
    #bl = np.empty(l.shape, dtype='uint8')
    bl = l >= o
    s = smooth(bl, N) 
    print s.max(), s.min()
    imsave('smoothed_lena.png', s)
    imsave('lena.png', bl)

def __test_circle():
    print "> Testing circle"
    from scipy.misc import imsave
    y, x = np.ogrid[0: 500, 0: 500]
    cx, cy = 250, 250
    r = 200
    ball = (y - cy)**2 + (x - cx)**2 <= r**2
    smoothed_ball = smooth(ball, N)
    imsave('ball.png', ball)
    imsave('smoothed_ball.png', smoothed_ball)

def main():
    __test_lena()
    __test_circle()
    

if __name__ == '__main__':
    main()
