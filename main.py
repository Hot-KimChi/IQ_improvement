## Image enhancement sequence
## 원본 이미지 생성 또는 불러오기:
# 초음파 이미지를 생성하거나 불러옵니다.

# 이미지 세그멘테이션 수행:
# 이미지 세그멘테이션 모델을 사용하여 원본 이미지의 객체 구조를 예측합니다.

# 노이즈 추가:
# 이미지 세그멘테이션 이후에 노이즈를 추가합니다.
# 이미지 세그멘테이션 결과를 보존하면서 노이즈를 추가해야 합니다.

# 모델 학습 및 평가:
# 세그멘테이션된 이미지와 노이즈가 추가된 이미지를 사용하여 모델을 학습하고 평가합니다.

## Import and Ready
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import img_to_array, load_img

AUTOTUNE = tf.data.AUTOTUNE


## Download the training dataset
def download_imgs():
    div2k_data =tfds.image.Div2k(config="bicubic_x4")
    div2k_data.download_and_prepare()
    
    # Taking train data from div2k_data object
    train = div2k_data.as_dataset(split="train", as_supervised=True)
    train_cache = train.cache()
    # Validation data
    val = div2k_data.as_dataset(split="validation", as_supervised=True)
    val_cache = val.cache()
    
    return train_cache, val_cache


class Dataset_Object:
    
    def __init__(self, dataset_cache, training=True):
        



# 노이즈를 추가할 원본 이미지 생성
def generate_original_image():
    # 여기에서 원본 이미지를 생성하는 코드 추가
    # 예를 들어, 이미지를 불러오거나 생성하는 함수를 사용
    # 여기에서는 간단히 테스트 이미지를 사용합니다.
    img_path = 'path_to_your_image.jpg'
    img = load_img(img_path, target_size=(224, 224))
    original_image = img_to_array(img)
    original_image = np.expand_dims(original_image, axis=0)

    return original_image

# 원본 이미지에 노이즈 추가
def add_noise_to_image(original_image, noise_factor=0.2):
    noise = np.random.normal(loc=0, scale=noise_factor, size=original_image.shape)
    noisy_image = original_image + noise

    return noisy_image


if __name__ == '__main__':
    download_imgs()
    generate_original_image()


# # 데이터셋 생성
# def create_dataset(num_samples=1000):
#     original_images = []
#     noisy_images = []

#     for _ in range(num_samples):
#         original_image = generate_original_image()
#         noisy_image = add_noise_to_image(original_image)

#         original_images.append(original_image)
#         noisy_images.append(noisy_image)

#     return np.array(original_images), np.array(noisy_images)

# # 데이터 전처리
# def preprocess_data(original_images, noisy_images):
#     # MobileNetV2의 입력 형식에 맞게 이미지 크기를 조정하고 정규화
#     original_images = tf.keras.applications.mobilenet_v2.preprocess_input(original_images)
#     noisy_images = tf.keras.applications.mobilenet_v2.preprocess_input(noisy_images)

#     return original_images, noisy_images

# # 모델 정의 (MobileNetV2 기반)
# def build_model(input_shape):
#     base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')

#     # MobileNetV2의 중간층에서 feature를 추출
#     base_model.trainable = False

#     model = models.Sequential([
#         base_model,
#         layers.GlobalAveragePooling2D(),
#         layers.Dense(1024, activation='relu'),
#         layers.Dense(input_shape[0] * input_shape[1] * input_shape[2], activation='sigmoid'),  # 픽셀 당 1개의 값을 예측
#         layers.Reshape((input_shape[0], input_shape[1], input_shape[2]))
#     ])

#     model.compile(optimizer='adam', loss='mse')

#     return model

# # 학습 및 평가
# def train_and_evaluate_model(original_images, noisy_images, epochs=10):
#     input_shape = original_images[0].shape

#     # 모델 빌드
#     model = build_model(input_shape)

#     # 데이터 전처리
#     original_images, noisy_images = preprocess_data(original_images, noisy_images)

#     # 모델 학습
#     model.fit(noisy_images, original_images, epochs=epochs, batch_size=32, shuffle=True)

#     # 예측
#     predicted_images = model.predict(noisy_images)

#     # 결과 시각화
#     visualize_results(original_images, noisy_images, predicted_images)

# # 결과 시각화
# def visualize_results(original_images, noisy_images, predicted_images):
#     plt.figure(figsize=(10, 7))

#     for i in range(5):  # 5개의 샘플만 시각화
#         plt.subplot(3, 5, i + 1)
#         plt.imshow(original_images[i])
#         plt.title('Original')

#         plt.subplot(3, 5, i + 6)
#         plt.imshow(noisy_images[i])
#         plt.title('Noisy')

#         plt.subplot(3, 5, i + 11)
#         plt.imshow(predicted_images[i])
#         plt.title('Predicted')

#     plt.show()

# # 데이터셋 생성
# original_images, noisy_images = create_dataset(num_samples=1000)

# # 모델 학습 및 평가
# train_and_evaluate_model(original_images, noisy_images, epochs=10)

