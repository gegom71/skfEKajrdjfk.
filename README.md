# skfEKajrdjfk.(MNIST TSET)


그냥 이사이트 들어가서 쓰셈
https://colab.research.google.com/drive/1MJRwJptHBE9W1KaQWtMJnp__F3KnTBCi?usp=drive_link





혹시모를 코드 작성도 존재
















# 텐서플로 다운받아.
!pip install tensorflow keras numpy pillow

# 적용
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import IPython.display as display
from google.colab import files

# MNIST 데이터셋 로드 및 전처리
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

# 간단한 CNN 모델 정의 및 훈련
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 모델 훈련 (빠른 테스트를 위해 epoch를 1(2_)로 설정)
model.fit(x_train, y_train, epochs=2, batch_size=64, validation_data=(x_test, y_test))

# 이미지 파일 업로드 (120x120 사이즈를 추천)
print("이미지 파일(28x28 회색조)을 업로드하세요:")
uploaded_image = files.upload()
for filename in uploaded_image:
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        img_path = filename
        break
else:
    print("No image file uploaded.")
    exit()

# 이미지 전처리
img = Image.open(img_path).convert('L')  # 회색조로 변환
img = img.resize((28, 28))  # 28x28로 리사이즈
img_arr = np.array(img)
img_arr = img_arr[:, :, np.newaxis]  # 채널 차원 추가
img_arr = img_arr / 255.0  # 픽셀 값 정규화
img_arr = img_arr[np.newaxis, :, :, :]  # 배치 차원 추가

# 예측
prediction = model.predict(img_arr)
predicted_digit = np.argmax(prediction)

# 결과 표시, 이미지 표시
print(f"예측된 숫자: {predicted_digit}")
# 이미지 표시
display.display(img)

print("확률 분포:")
for i in range(10):
    print(f"{i}: {prediction[0][i]:.2f}")
