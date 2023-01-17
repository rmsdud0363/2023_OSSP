# 2023_OSSP_Test

# 라이브러리 import
# PIL: Python Imaging Library
from PIL import Image
# os: 운영체제 관련 인터페이스 제공 라이브러리
import os
# glob: 파일들의 리스트를 뽑을 때 사용하는 라이브러리
import glob
# numpy: 행렬이나 다차원 배열을 쉽게 처리할 수 있도록 지원하는 라이브러리
import numpy as np
# sklearn.model_selection: 기계 학습 모델 및 통계 모델링을 구현하기 위한 Python 도구 키트
# train_test_split: 데이터를 training data set과 test data set으로 나눔
from sklearn.model_selection import train_test_split
# keras: 오픈소스 신경망 라이브러리
# Sequential: 순차적으로 레이어를 쌓아주는 케라스 라이브러리
from keras.models import Sequential
# Conv2D, MaxPooling2D, Dense, Flatten, Dropout: 신경망 층을 쌓기 위한 라이브러리
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
# EarlyStopping: 모델이 더 이상 학습을 못할 경우, 학습 도중 미리 학습을 종료시키는 콜백함수
# ModelCheckpoint: 모델을 저장할 때 사용되는 콜백함수
from keras.callbacks import EarlyStopping, ModelCheckpoint
# matplotlib: 여러 가지 그래프를 그릴 수 있는 함수들이 들어있는 라이브러리
import matplotlib.pyplot as plt
# tensorflow: 머신러닝을 위한 오픈소스 라이브러리
import tensorflow as tf
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# 이미지 경로 및 변수 지정
# data set 불러오기
data_dir = 'C:/Users/rmsdu/OneDrive/문서/GitHub/2023_OSSP/2023_OSSP_Data/Train'
# data set category 설정 (categories 리스트는 최종 결과를 반환할 때 사용)
categories = ['coca', 'fanta', 'letsbee', 'pocari', 'sprite', 'tejava']
# data set에 category 개수(길이)를 nb_classes에 저장
nb_classes = len(categories)

# 이미지의 크기를 모두 통일
image_w = 64
image_h = 64

# X, Y는 데이터와 라벨을 저장하기 위해 만든 리스트
X = []
Y = []

# 이미지 전처리 1
# 경로에서 불러 온 이미지를 일일이 리사이징
# for idx, cat in enumerate(): 순서가 있는 자료형을 입력으로 받아 인덱스 값을 포함하는 튜플로 만들어줌
# idx = 인덱스 값/ 0부터 시작, cat = categories 리스트 원소
for idx, cat in enumerate(categories):

    # one-hot encoding
    # i = range(nb_classes)
    # range(nb_classes)로 설정하여 data set category의 수가 변경되어도 상관이 없음
    label = [0 for i in range(nb_classes)]
    label[idx] = 1

    # data set category 별 image 경로 지정
    image_dir = data_dir + "/" + cat
    # '.png'인 이미지만 리스트로 뽑아오기
    files = glob.glob(image_dir + "/*.jpg")
    # category 별 파일 길이 출력하여 확인

    for i, f in enumerate(files):
        # Image.open(f: 이미지 파일의 경로): 이미지 파일 열기
        img = Image.open(f)
        # img.convert("RGB"): 열어 놓은 이미지 파일을 RGB로 변환
        img = img.convert("RGB")
        # img.resize((image_w, image_h)): 64x64 픽셀로 리사이즈
        img = img.resize((image_w, image_h))
        # Numpy 배열 데이터로 변환
        # np.asarray(): PIL Image를 NumPy array로 변환해주는 함수
        data = np.asarray(img)
        # X 리스트에 data 정보를 요소로 추가
        X.append(data)
        # Y 리스트에 label 정보를 요소로 추가
        Y.append(label)

        if i % 700 == 0:
            print(cat, " : ", f)

# 리스트 X, Y를 배열로 변환 후 저장
X = np.array(X)
Y = np.array(Y)

# 데이터 불러오기
# data set을 순차적으로 training data set과 test data set으로 분할
# validation_data=(X_test, Y_test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

# data set 배열을 npy 파일로 저장
np.save("C:/Users/rmsdu/OneDrive/문서/GitHub/2023_OSSP/NPY_File/X_train.npy", X_train)
np.save("C:/Users/rmsdu/OneDrive/문서/GitHub/2023_OSSP/NPY_File/X_test.npy", X_test)
np.save("C:/Users/rmsdu/OneDrive/문서/GitHub/2023_OSSP/NPY_File/Y_train.npy", Y_train)
np.save("C:/Users/rmsdu/OneDrive/문서/GitHub/2023_OSSP/NPY_File/Y_test.npy", Y_test)

# ConfigProto: 연산 방식(CPU/GPU)을 설정하는 기능의 함수
config = tf.compat.v1.ConfigProto()

# GPU 메모리 관리
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# npy 파일을 로드
X_train = np.load("C:/Users/rmsdu/OneDrive/문서/GitHub/2023_OSSP/NPY_File/X_train.npy")
X_test = np.load("C:/Users/rmsdu/OneDrive/문서/GitHub/2023_OSSP/NPY_File/X_test.npy")
Y_train = np.load("C:/Users/rmsdu/OneDrive/문서/GitHub/2023_OSSP/NPY_File/Y_train.npy")
Y_test = np.load("C:/Users/rmsdu/OneDrive/문서/GitHub/2023_OSSP/NPY_File/Y_test.npy")

print(X_train.shape)
print(X_train.shape[0])

# test set category 설정
categories = ['coca_T', 'fanta_T', 'letsbee_T', 'pocari_T', 'sprite_T', 'tejava_T']
nb_classes = len(categories)

#일반화
X_train = X_train.astype(float) / 255
X_test = X_test.astype(float) / 255

# 순차적으로 모델을 쌓기

# 합성곱 신경망 구성
model = Sequential()
model.add(Conv2D(32, (3,3), padding="same", input_shape=X_train.shape[1:], activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))
# 학습 방식 환경 설정
# optimizer: 훈련과정 설정, 최적화 알고리즘 설정
# loss: 모델 최적화 목적 함수
# metrics: 훈련을 모니터링 하기 위해 사용/ accuracy: 분류에 사용
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], run_eagerly=True)
model_dir = './model'

# 폴더가 존재하지 않으면
if not os.path.exists(model_dir):
    # model_dir 이름의 폴더 생성
    os.mkdir(model_dir)

# 모델 경로 지정
model_path = model_dir + '/multi_img_classification.model'
# 콜백 함수
# verbose: 해당 함수의 진행 사항의 출력 여부
# save_best_only: 모델의 정확도가 최고값을 갱신했을 때만 저장
checkpoint = ModelCheckpoint(filepath=model_path , monitor='val_loss', verbose=1, save_best_only=True)
# overfitting 방지
early_stopping = EarlyStopping(monitor='val_loss', patience=6)

model.summary()

# 모델 학습(실행)
history = model.fit(X_train, Y_train, batch_size=32, epochs=50, validation_data=(X_test, Y_test), callbacks=[checkpoint, early_stopping])
# evaluate: 모델에 대한 정확도 평가
# X_test: 테스트 데이터/ Y_test: 지도 학습에서 '레이블 테스트 데이터'
print("정확도 : %.4f" % (model.evaluate(X_test, Y_test)[1]))

# validation_data_set과 training_data_set의 오차를 저장
Y_vloss = history.history['val_loss']
Y_loss = history.history['loss']

# 그래프로 표현
x_len = np.arange(len(Y_loss))

plt.plot(x_len, Y_vloss, marker='.', c='red', label='val_set_loss')
plt.plot(x_len, Y_loss, marker='.', c='blue', label='train_set_loss')

# 그리드, 레이블 표시
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.show()

# Keras 모델을 저장하고 불러오는 라이브러리
from keras.models import load_model
tf.data.experimental.enable_debug_mode()

# test data set 불러오기
data_dir = "C:/Users/rmsdu/OneDrive/문서/GitHub/2023_OSSP/2023_OSSP_Data/Test"
image_w = 64
image_h = 64

# 데이터와 파일 이름을 저장할 리스트
X = []
filenames = []

# 이미지 전처리 2 -> 위의 코드와 동일
for idx, cat in enumerate(categories):
    label = [0 for i in range(nb_classes)]
    label[idx] = 1

    image_dir = data_dir + "/" + cat
    files = glob.glob(image_dir + "/*.jpg")

    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)
        filenames.append(f)
        X.append(data)

# 데이터 리스트를 배열로 전환
X = np.array(X)
# 모델 불러오기
model = load_model('./model/multi_img_classification.model')

# 테스트 데이터 X에 대해 예측하기
prediction = model.predict(X)
# numpy float 출력 옵션 변경
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
cnt = 0


for i in prediction:
    pre_ans = i.argmax()  # 예측 레이블
    print(i)
    print(pre_ans)
    pre_ans_str = ''
    if pre_ans == 0: pre_ans_str = "코카콜라"
    elif pre_ans == 1: pre_ans_str = "환타"
    elif pre_ans == 2: pre_ans_str = "레쓰비"
    elif pre_ans == 3: pre_ans_str = "포카리"
    elif pre_ans == 4: pre_ans_str = "스프라이트"
    elif pre_ans == 5: pre_ans_str = "데자와"

    else: pre_ans_str = "게"
    if i[0] >= 0.8: print("해당 "+filenames[cnt].split("\\")[1]+"이미지는 "+pre_ans_str+"로 추정됩니다.")
    if i[1] >= 0.8: print("해당 "+filenames[cnt].split("\\")[1]+"이미지는 "+pre_ans_str+"으로 추정됩니다.")
    if i[2] >= 0.8: print("해당 "+filenames[cnt].split("\\")[1]+"이미지는 "+pre_ans_str+"로 추정됩니다.")
    if i[3] >= 0.8: print("해당 " + filenames[cnt].split("\\")[1] + "이미지는 " + pre_ans_str + "로 추정됩니다.")
    if i[4] >= 0.8: print("해당 " + filenames[cnt].split("\\")[1] + "이미지는 " + pre_ans_str + "로 추정됩니다.")
    if i[5] >= 0.8: print("해당 " + filenames[cnt].split("\\")[1] + "이미지는 " + pre_ans_str + "로 추정됩니다.")
    cnt += 1
