import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
import os
from tqdm import tqdm
import pickle
from numpy.linalg import norm

# Load Pretrained Model
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# Feature Extraction Function
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)

    result = model.predict(preprocessed_img, verbose=0).flatten()
    normalized_result = result / norm(result)

    return normalized_result

# Get Image File Paths
filenames = []
image_folder = "images"

for file in os.listdir(image_folder):
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        filenames.append(os.path.join(image_folder, file))

# Extract Features
feature_list = []

for file in tqdm(filenames):
    feature_list.append(extract_features(file, model))

# Save Files
pickle.dump(feature_list, open("embeddings.pkl", "wb"))
pickle.dump(filenames, open("filenames.pkl", "wb"))

print("✅ embeddings.pkl and filenames.pkl saved successfully!")
