import sys
import numpy as np
import binvol_cy

def test(n):
    m_no_smoothed = np.memmap('no_smoothed_0051.dat', shape=(127, 532, 532),
                              dtype='bool', mode='r')
    smoothed = np.memmap('smoothed_0051_%sx.dat' % n, shape=(127, 532, 532),
                         dtype='float64', mode='w+')
    print 'writing result to', smoothed.filename
    smoothed[:] = binvol_cy.smooth(m_no_smoothed, n)
    smoothed.flush()
    smoothed.close()

def main():
    n = int(sys.argv[1])
    test(n)

if __name__ == '__main__':
    main()
