import os
from keras.utils.visualize_util import plot
filepath=os.path.split(os.path.realpath(__file__))[0]+'/model.png'
def model_gragh(model,filepath=None):
    if filepath==None:
        filepath=os.path.join(os.path.split(os.path.realpath(__file__))[0],'model.png')
    plot(model, to_file=filepath)