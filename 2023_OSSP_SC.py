# 2023_OSSP_SourceCode

# 코드를 돌리기 위한 라이브러리
# 이미지 분석 및 처리 라이브러리
from PIL import Image
# os: .csv로 끝나는 파일명, glob: 파일들의 리스트, numpy: 행렬, 다차원 배열
import os, glob, numpy as np
from sklearn.model_selection import train_test_split
import os, glob, numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import tensorflow as tf


data_dir = 'C:/Users/rmsdu/OneDrive/문서/GitHub/2023_OSSP/2023_OSSP_Data/Train Set'
categories = ["Coca", "Sprite", "Pocari"]
nb_classes = len(categories)

image_w = 64
image_h = 64

pixels = image_h * image_w * 3

X = []
y = []

for idx, cat in enumerate(categories):

    # one-hot 돌리기.
    label = [0 for i in range(nb_classes)]
    label[idx] = 1

    image_dir = data_dir + "/" + cat
    files = glob.glob(image_dir + "/*.png")
    print(cat, " 파일 길이 : ", len(files))
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)

        X.append(data)
        y.append(label)

        if i % 700 == 0:
            print(cat, " : ", f)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y)
xy = (X_train, X_test, y_train, y_test)
np.save("C:/Users/rmsdu/OneDrive/문서/GitHub/2023_OSSP/multi_image_data.npy", xy)

print("ok", len(y))

config = tf.compat.v1.ConfigProto()
"""
    config  :   
                tensorflow 2.x 버전 업데이트 이후 
                tf.ConfigProto() -> tf.compat.v1.ConfigProto()
"""
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
"""
    session  :   
                tensorflow 2.x 버전 업데이트 이후 
                tf.session() -> tf.compat.v1.session()
"""
