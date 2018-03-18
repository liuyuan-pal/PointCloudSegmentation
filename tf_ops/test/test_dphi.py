import numpy as np
from sklearn.cluster import KMeans
from draw_util import output_points

center_num=5

pts=np.random.uniform(-1.0,1.0,[100000,3])
pts/=np.sqrt(np.sum(pts**2,axis=1,keepdims=True)+1e-6)
kmeans=KMeans(center_num)
labels=kmeans.fit_predict(pts)
centers=[]
for i in xrange(center_num):
    centers.append(np.mean(pts[labels==i],axis=0))

centers=np.asarray(centers)

colors=np.random.randint(0,256,[center_num,3])

ang1=-np.arctan2(centers[0,0],centers[0,1])
cosv,sinv=np.cos(ang1),np.sin(ang1)
m=np.asarray([[cosv ,-sinv,0],
              [sinv,cosv,0],
              [0    ,0   ,1]],dtype=np.float64)
centers,pts=np.dot(centers,m),np.dot(pts,m)
print centers[0]

ang2=-(np.pi/2-np.arctan2(centers[0,2],centers[0,1]))
cosv,sinv=np.cos(ang2),np.sin(ang2)
m=np.asarray([[1,0,0],
              [0,cosv,-sinv],
              [0,sinv,cosv]],dtype=np.float64)
centers,pts=np.dot(centers,m),np.dot(pts,m)
print centers[0]

output_points('/home/pal/tmp/points.txt',pts,colors[labels,:])
output_points('/home/pal/tmp/centers.txt',centers,colors)
