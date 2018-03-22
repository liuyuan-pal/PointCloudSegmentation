import numpy as np
from draw_util import output_points
import cPickle
from cached.nyu_structured_classes import get_structure_classes

def save_pkl(fn,obj):
    with open(fn,'wb') as f:
        cPickle.dump(obj,f,2)

def depth2points(depths, rgbs, inv_kd):
    hs,ws=np.meshgrid(np.arange(depths.shape[0]),np.arange(depths.shape[1]))
    hs=np.expand_dims(hs.transpose(),axis=2)
    ws=np.expand_dims(ws.transpose(),axis=2)
    ones=np.ones(hs.shape,dtype=np.float32)
    hwos=np.concatenate([hs,ws,ones],axis=2)
    xyzs=inv_kd.dot(np.expand_dims      # [3,3] dot [h,w,3,1]
                    (np.expand_dims(depths,axis=2)*hwos,axis=3))
    xyzs=np.squeeze(xyzs,axis=3)
    xyzs=xyzs.transpose([1,2,0])

    xyzs=np.reshape(xyzs,[-1,3])
    rgbs=np.reshape(rgbs,[-1,3])

    return np.asarray(np.concatenate([xyzs,rgbs],axis=1),np.float32)

if __name__=="__main__":
    import h5py
    f=h5py.File('/home/pal/data/NYUv2/nyu_depth_v2_labeled.mat')
    depths=f['depths'][:]
    images=f['images'][:]
    labels=f['labels'][:]
    names = [u''.join(unichr(c) for c in f[obj_ref]) for obj_ref in f['names'][0]]

    classes=get_structure_classes()
    f.close()

    k1_rgb = 2.0796615318809061e-01
    k2_rgb = -5.8613825163911781e-01
    p1_rgb = 7.2231363135888329e-04
    p2_rgb = 1.0479627195765181e-03
    k3_rgb = 4.9856986684705107e-01
    dis_coeff_rgb=np.asarray([k1_rgb,k2_rgb,p1_rgb,p2_rgb,k3_rgb])

    k1_d = -9.9897236553084481e-02
    k2_d = 3.9065324602765344e-01
    p1_d = 1.9290592870229277e-03
    p2_d = -1.9422022475975055e-03
    k3_d = -5.1031725053400578e-01
    dis_coeff_d=np.asarray([k1_d,k2_d,p1_d,p2_d,k3_d])

    fx_rgb = 5.1885790117450188e+02
    fy_rgb = 5.1946961112127485e+02
    cx_rgb = 3.2558244941119034e+02
    cy_rgb = 2.5373616633400465e+02

    fx_d = 5.8262448167737955e+02
    fy_d = 5.8269103270988637e+02
    cx_d = 3.1304475870804731e+02
    cy_d = 2.3844389626620386e+02
    K=[[fx_rgb,0,cx_rgb],
       [0,fy_rgb,cy_rgb],
       [0,0,1]]
    Kd=[[fx_d,0,cx_d],
       [0,fy_d,cy_d],
       [0,0,1]]
    K=np.asarray(K,np.float32)
    Kd=np.asarray(Kd,np.float32)
    inv_kd=np.linalg.inv(Kd)
    inv_k=np.linalg.inv(K)
    print 'copy done'

    views=[]
    vid,count=0,0
    for i in xrange(depths.shape[0]):
        img=images[i].transpose([1,2,0])
        dep=depths[i]

        points=depth2points(dep,img,inv_kd)
        views.append([points,np.asarray(labels[i].flatten(),np.int32)])

        count+=1
        print 'max label {}'.format(np.max(labels[i]))

        if count>200:
            save_pkl('/home/pal/data/NYUv2/batch_{}.pkl'.format(vid),views)
            views=[]
            count=0
            vid+=1

    if count > 0:
        save_pkl('/home/pal/data/NYUv2/batch_{}.pkl'.format(vid), views)

