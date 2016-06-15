import numpy as np
import cv2,os
from matplotlib import pyplot as plt

import PR2016
from utils import subimshow_listitems

#pre settings
datadir = '/home/suber/Downloads/pr2016'
img_size=70
PCA_sign=True
PCA_components=20
if PCA_sign:
	from sklearn.decomposition import PCA

#load respect dataset
(X_train, y_train), (X_test, y_test) = PR2016.load_data(img_size)

#preprocessing
def preprocess(X):
	n=X.shape[0]
	new_X=np.zeros((n,(img_size**2)),dtype=X.dtype)
	for i in xrange(n):
		img=X[i].transpose(1,2,0)
		new_X[i] = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).flatten()
	return new_X

##Now we preprocess the data and shape should be converted to (10846,(img_size**2))
print 'begin preprocessing...' 
X_train=preprocess(X_train)
X_test=preprocess(X_test)
print 'finish preprocessing'

#PCA
def showpca(pca):
	comlist=pca.components_.reshape(PCA_components,img_size,img_size).tolist()
	fig=subimshow_listitems(comlist)
	return fig

if PCA_sign:
	print 'begin PCA...'
	pca = PCA(n_components=PCA_components)
	X_train = pca.fit_transform(X_train)
	X_test = pca.transform(X_test)
	print 'finish PCA'
	fig=showpca(pca)



# Initiate kNN, train the data, then test it with test data for k=5
knn = cv2.KNearest()
knn.train(X_train.astype(np.float32),y_train.astype('int64'))
ret,result,neighbours,dist = knn.find_nearest(X_test.astype(np.float32),k=5)

# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
matches = result==y_test.astype('int64')
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.size
print "PCA+KNN accracy %.2f"%accuracy+'%'


svm_params = dict( kernel_type = cv2.SVM_LINEAR,
                    svm_type = cv2.SVM_C_SVC,
                    C=1, gamma=0.5 )

# Initiate kNN, train the data, then test it with test data
svm = cv2.SVM()
svm.train(X_train.astype(np.float32),y_train.astype(np.float32), params=svm_params)
svm.save(os.path.join(datadir,'svm_data.dat'))
result = svm.predict_all(X_test.astype(np.float32))

# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
matches = result.astype('int64')==y_test.astype('int64')
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.shape[0]
print "PCA+SVM accracy %.2f"%accuracy+'%'

"""from sklearn import svm
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X_train.astype(np.float32).tolist(), y_train.astype(np.int).flatten().tolist()) 
result=clf.predict(X_test.astype(np.float32))

# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
matches = result==y_test.flatten()
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.shape[0]
print "PCA+SVM accracy %.2f"%accuracy+'%'"""

#========================================================================
'''SZ=20
bin_n = 16 # Number of bins

svm_params = dict( kernel_type = cv2.SVM_LINEAR,
                    svm_type = cv2.SVM_C_SVC,
                    C=2.67, gamma=5.383 )

affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img

def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist

img = cv2.imread('digits.png',0)

cells = [np.hsplit(row,100) for row in np.vsplit(img,50)]

# First half is trainData, remaining is testData
train_cells = [ i[:50] for i in cells ]
test_cells = [ i[50:] for i in cells]

######     Now training      ########################

deskewed = [map(deskew,row) for row in train_cells]
hogdata = [map(hog,row) for row in deskewed]
trainData = np.float32(hogdata).reshape(-1,64)
responses = np.float32(np.repeat(np.arange(10),250)[:,np.newaxis])

svm = cv2.SVM()
svm.train(trainData,responses, params=svm_params)
svm.save('svm_data.dat')

######     Now testing      ########################

deskewed = [map(deskew,row) for row in test_cells]
hogdata = [map(hog,row) for row in deskewed]
testData = np.float32(hogdata).reshape(-1,bin_n*4)
result = svm.predict_all(testData)

#######   Check Accuracy   ########################
mask = result==responses
correct = np.count_nonzero(mask)
print correct*100.0/result.size'''