import os
from pathlib import Path
import multiprocessing as mp

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from skimage.data import imread
from sklearn.ensemble import RandomForestClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "./"]).decode("utf8"))

input_path = Path('./')
train_path = input_path / 'train'
test_path = input_path / 'test'

cameras = os.listdir(train_path)

train_images = []
for camera in cameras:
    for fname in sorted(os.listdir(train_path / camera)):
        train_images.append((camera, fname))

train = pd.DataFrame(train_images, columns=['camera', 'fname'])
print(train.shape)
train.sample(5)

test_images = []
for fname in sorted(os.listdir(test_path)):
    test_images.append(fname)

test = pd.DataFrame(test_images, columns=['fname'])
print(test.shape)
test.head(5)

def color_stats(q, iolock):
    
    while True:
        
        img_path = q.get()
        if img_path is None:
            break
            
        if type(img_path) is tuple:
            img = imread(train_path / img_path[0] / img_path[1])
            key = img_path[1]
        else:
            img = imread(test_path / img_path)
            key = img_path

        # Some images read return info in a 2nd dim. We only want the first dim.
        if img.shape == (2,):
            img = img[0]

        color_info[key] = (img[:, :, 0].mean(), img[:, :, 1].mean(), img[:, :, 2].mean(),
                           img[:, :, 0].std(),  img[:, :, 1].std(),  img[:, :, 2].std())

cols = ['a0', 'a1', 'a2', 's0', 's1', 's2']

for col in cols:
    train[col] = None
    test[col] = None

NCORE = 8

color_info = mp.Manager().dict()

# Using a queue since the image read is a bottleneck
q = mp.Queue(maxsize=NCORE)
iolock = mp.Lock()
pool = mp.Pool(NCORE, initializer=color_stats, initargs=(q, iolock))

for i in train_images:
    q.put(i)  # blocks until q below its max size

for i in test_images:
    q.put(i)  # blocks until q below its max size
    
# tell workers we're done
for _ in range(NCORE):  
    q.put(None)
pool.close()
pool.join()

color_info = dict(color_info)

for n, col in enumerate(cols):
    train[col] = train['fname'].apply(lambda x: color_info[x][n])
    test[col] = test['fname'].apply(lambda x: color_info[x][n])
    
train.sample(5)

y = train['camera'].values
X_train = train[cols].values
X_test = test[cols].values
clf = RandomForestClassifier(n_estimators=200)
clf.fit(X_train, y)

y_pred = clf.predict(X_test)
print(y_pred)
clueless = pd.read_csv(input_path / 'sample_submission.csv', index_col='fname')
clueless['camera'] = y_pred
clueless.to_csv('clueless.csv') 