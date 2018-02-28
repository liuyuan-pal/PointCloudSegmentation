import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import itertools



def output_activation(feature, filename, dim, pts):
    pt_num=pts.shape[0]
    # indices = np.argsort(-feature[:, dim])
    max_feature_val = np.max(feature[:, dim])
    min_feature_val = np.min(feature[:, dim])

    color_count = np.zeros(256,np.float32)
    for i in xrange(pt_num):
        this_color = (feature[i, dim]-min_feature_val) / (max_feature_val-min_feature_val) *255
        color_count[int(this_color)]+=1

    for i in range(1,256):
        color_count[i]+=color_count[i-1]

    color_count/=color_count[-1]

    color = np.random.uniform(0, 1, [3])
    color = color/np.sqrt(np.sum(color**2))
    with open(filename, 'w') as f:
        for i in xrange(pt_num):
                this_color = color_count[int((feature[i, dim]-min_feature_val) /
                                             (max_feature_val-min_feature_val)*255)]*color
                this_color = np.asarray(this_color*255, np.int)
                f.write('{} {} {} {} {} {}\n'.format(
                    pts[i, 0], pts[i, 1], pts[i, 2],
                    this_color[0], this_color[1], this_color[2]))


def output_activation_distribution(feature,dim,filename):
    import matplotlib.pyplot as plt
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.hist(feature[:,dim].flatten(),bins=100)
    fig.savefig(filename)
    plt.close(fig)


def output_points(filename,pts,colors=None):
    has_color=pts.shape[1]>=6
    with open(filename, 'w') as f:
        for i,pt in enumerate(pts):
            if colors is None:
                if has_color:
                    f.write('{} {} {} {} {} {}\n'.format(pt[0],pt[1],pt[2],int(pt[3]),int(pt[4]),int(pt[5])))
                else:
                    f.write('{} {} {}\n'.format(pt[0],pt[1],pt[2]))

            else:
                if len(colors.shape)==2:
                    f.write('{} {} {} {} {} {}\n'.format(pt[0],pt[1],pt[2],colors[i,0],colors[i,1],colors[i,2]))
                else:
                    f.write('{} {} {} {} {} {}\n'.format(pt[0],pt[1],pt[2],colors[0],colors[1],colors[2]))


def plot_confusion_matrix(preds, labels, names,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          save_path='test_result/'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm=confusion_matrix(labels,preds,range(len(names)))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(1,figsize=(10, 8), dpi=80)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path+'confusion_matrix.png')
    plt.close()
