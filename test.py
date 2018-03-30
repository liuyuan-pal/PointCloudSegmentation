import numpy as np


def iter_compute(a,b):
    t,_,_=a.shape
    t,_,o=b.shape
    out=np.empty([t,o])
    for i in xrange(t):
        out[i]=np.matmul(a[i,0,:],b[i])

    return out


a=np.random.randint(0,255,[3,1,2])
b=np.random.randint(0,255,[3,2,4])

print iter_compute(a,b)
print np.matmul(a,b)[:,0,:]