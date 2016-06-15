import os
from keras.utils.visualize_util import plot
from matplotlib import pyplot as plt
import numpy as np
filepath=os.path.split(os.path.realpath(__file__))[0]+'/model.png'
def model_gragh(model,filepath=None):
    if filepath==None:
        filepath=os.path.join(os.path.split(os.path.realpath(__file__))[0],'model.png')
    plot(model, to_file=filepath)

def subimshow_listitems(listI,name='pca figure',color='gray'):
	n=len(listI)
	height=int(np.floor(np.sqrt(n)))
	width=int(np.ceil(float(n)/height))
	fig, axes = plt.subplots(height, width, figsize=(12, 6),
                         subplot_kw={'xticks': [], 'yticks': []})
	fig.subplots_adjust(hspace=0.05, wspace=0.05)
	fig.suptitle(name)
	for i in xrange(n):
	    axes.flat[i].imshow(listI[i],interpolation='none',cmap=color)
	plt.show()
	return fig