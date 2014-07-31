import sys

from pylab import*
filename =sys.argv[1]
a = np.fromfile(filename,dtype=np.int8,count=41943040)

a.tofile('ipfb_decreased.dat', sep="",format="%s")
print a.shape
