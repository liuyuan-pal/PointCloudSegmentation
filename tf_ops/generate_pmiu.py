import os
path = os.path.split(os.path.realpath(__file__))[0]
import sys
sys.path.append(path)
sys.path.append(path+'/test')
import numpy as np
from sklearn.cluster import KMeans
from draw_util import output_points


def generate_anchor(center_num=5):

    if os.path.exists('cached/centers.txt'):
        with open('cached/centers.txt','r') as f:
            centers=[]
            for line in f.readlines():
                line=line.strip('\n')
                subline=line.split(' ')
                centers.append([float(subline[0]),float(subline[1]),float(subline[2])])

            centers=np.asarray(centers,dtype=np.float32)
        if centers.shape[0]==center_num:
            return centers.transpose()

    pts=np.random.uniform(-1.0,1.0,[100000,3])
    pts/=np.sqrt(np.sum(pts**2,axis=1,keepdims=True)+1e-6)
    kmeans=KMeans(center_num)
    labels=kmeans.fit_predict(pts)
    centers=[]
    for i in xrange(center_num):
        centers.append(np.mean(pts[labels==i],axis=0))

    centers=np.asarray(centers)

    ang1=-np.arctan2(centers[0,0],centers[0,1])
    cosv,sinv=np.cos(ang1),np.sin(ang1)
    m=np.asarray([[cosv ,-sinv,0],
                  [sinv,cosv,0],
                  [0    ,0   ,1]],dtype=np.float64)
    centers,pts=np.dot(centers,m),np.dot(pts,m)

    ang2=-(np.pi/2-np.arctan2(centers[0,2],centers[0,1]))
    cosv,sinv=np.cos(ang2),np.sin(ang2)
    m=np.asarray([[1,0,0],
                  [0,cosv,-sinv],
                  [0,sinv,cosv]],dtype=np.float64)
    centers,pts=np.dot(centers,m),np.dot(pts,m)

    output_points('cached/centers.txt',centers)

    return centers.transpose()