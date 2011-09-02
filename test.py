import sys
import numpy as np
import binvol_cy

def test_0051(n):
    m_no_smoothed = np.memmap('no_smoothed_0051.dat', shape=(127, 532, 532),
                              dtype='bool', mode='r')
    smoothed = np.memmap('smoothed_0051_%sx.dat' % n, shape=(127, 532, 532),
                         dtype='float64', mode='w+')
    print 'writing result to', smoothed.filename
    smoothed[:] = binvol_cy.smooth(m_no_smoothed, n)
    smoothed.flush()
    smoothed.close()

def test_bars(n):
    bars = np.zeros((500, 500, 500), dtype='bool')
    bars[5:495, 5:250, 5:250] = 1  # Bloco 1, vermelho (250x250)
    bars[5:495, 312:438, 62:188] = 1  # Bloco 2, verde (125x125)
    bars[5:495, 87:162, 337:412] = 1  # Bloco 3, azul (62x62)
    bars[5:495, 375, 375] = 1  # (1x1)
    bars[5:495, 492:494, 250:252] = 1  # (2x2)
    bars[5:495, 491:494, 491:494] = 1  # (3x3)
    bars[5:495, 250:254, 490:494] = 1  # (4x4)
    
    #no_smoothed = np.memmap('no_smoothed_bars.dat', shape=bars.shape,
                            #dtype='bool', mode='w+')
    #no_smoothed[:] = bars[:]

    smoothed = np.memmap('bars_%sx.dat' % n, shape=bars.shape,
                         dtype='float64', mode='w+')
    print 'writing result to', smoothed.filename
    smoothed[:] = binvol_cy.smooth(bars, n)
    smoothed.flush()
    smoothed.close()

def test_50ball(n):
    dim = 50
    z, x, y = np.ogrid[0:dim, 0:dim, 0:dim]
    r = 20
    c = dim / 50
    ball = (x - c)**2 + (y - c)**2 + (z - c)**2 <= r**2
    smoothed_ball = binvol_cy.smooth(ball, n)

def main():
    n = int(sys.argv[1])
    test_bars(n)

if __name__ == '__main__':
    main()
