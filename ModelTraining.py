import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, MaxPooling2D, Dropout,
    Flatten, Dense, Multiply, GlobalAveragePooling2D, Reshape
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tqdm import tqdm

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
CLASS_MAP = {
    0: "Mild Dementia",
    1: "Moderate Dementia",
    2: "Non Demented",
    3: "Very mild Dementia"
}


class MemorySafeDataLoader:
    def __init__(self, data_dirs):
        self.image_paths = []
        self.labels = []

        non_demented = []
        very_mild_demented = []
        mild_demented = []
        moderate_demented = []

        for dirname, _, filenames in os.walk(data_dirs['Non Demented']):
            non_demented.extend([os.path.join(dirname, f) for f in filenames])

        for dirname, _, filenames in os.walk(data_dirs['Very mild Dementia']):
            very_mild_demented.extend([os.path.join(dirname, f) for f in filenames])

        for dirname, _, filenames in os.walk(data_dirs['Mild Dementia']):
            mild_demented.extend([os.path.join(dirname, f) for f in filenames])

        for dirname, _, filenames in os.walk(data_dirs['Moderate Dementia']):
            moderate_demented.extend([os.path.join(dirname, f) for f in filenames])

        self.data_pairs = []
        self.data_pairs.extend([(p, 0) for p in non_demented])
        self.data_pairs.extend([(p, 1) for p in mild_demented])
        self.data_pairs.extend([(p, 2) for p in moderate_demented])
        self.data_pairs.extend([(p, 3) for p in very_mild_demented])

    def _process_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        return img.astype(np.float32) / 255.0

    def data_generator(self, indices=None):
        if indices is None:
            indices = np.arange(len(self.data_pairs))

        while True:
            np.random.shuffle(indices)
            for i in range(0, len(indices), BATCH_SIZE):
                batch_indices = indices[i:i + BATCH_SIZE]
                batch_images = []
                batch_labels = []

                for idx in batch_indices:
                    path, label = self.data_pairs[idx]
                    batch_images.append(self._process_image(path))
                    batch_labels.append(label)

                yield np.array(batch_images), np.array(batch_labels)

    def get_sample_data(self, num_samples=100):
        indices = np.random.choice(len(self.data_pairs), num_samples, replace=False)
        samples = [self._process_image(self.data_pairs[i][0]) for i in indices]
        labels = [self.data_pairs[i][1] for i in indices]
        return np.array(samples), np.array(labels)


class SEBlock(tf.keras.layers.Layer):
    def __init__(self, ratio=8, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        self.channel = input_shape[-1]
        self.global_avg_pool = GlobalAveragePooling2D()
        self.reshape = Reshape((1, 1, self.channel))
        self.fc1 = Dense(
            self.channel // self.ratio,
            activation='relu',
            kernel_initializer='he_normal',
            use_bias=False
        )
        self.fc2 = Dense(
            self.channel,
            activation='sigmoid',
            kernel_initializer='he_normal',
            use_bias=False
        )
        super(SEBlock, self).build(input_shape)

    def call(self, inputs):
        se_feature = self.global_avg_pool(inputs)
        se_feature = self.reshape(se_feature)
        se_feature = self.fc1(se_feature)
        se_feature = self.fc2(se_feature)
        return Multiply()([inputs, se_feature])

    def get_config(self):
        config = super().get_config()
        config.update({'ratio': self.ratio})
        return config


def build_original_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=2, padding='Same', input_shape=(128, 128, 3)))
    model.add(Conv2D(filters=32, kernel_size=2, padding='Same', activation='relu'))
    model.add(BatchNormalization())
    model.add(SEBlock())

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=2, padding='Same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=2, padding='Same', activation='relu'))
    model.add(BatchNormalization())
    model.add(SEBlock())

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=128, kernel_size=2, padding='Same', activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=2, padding='Same', activation='relu'))
    model.add(BatchNormalization())
    model.add(SEBlock())

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(4, activation='softmax'))

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_model():
    data_dirs = {
        'Non Demented': '/unzipped/Data/Non Demented',
        'Very mild Dementia': '/unzipped/Data/Very mild Dementia',
        'Mild Dementia': '/unzipped/Data/Mild Dementia',
        'Moderate Dementia': '/unzipped/Data/Moderate Dementia'
    }
    loader = MemorySafeDataLoader(data_dirs)

    # Train-Data split
    all_indices = np.arange(len(loader.data_pairs))
    train_idx, val_idx = train_test_split(all_indices, test_size=0.2, random_state=42)

    # create explainer
    train_gen = loader.data_generator(train_idx)
    val_gen = loader.data_generator(val_idx)

    # Create model
    model = build_original_model()

    # Train Define
    steps_per_epoch = len(train_idx) // BATCH_SIZE
    validation_steps = len(val_idx) // BATCH_SIZE

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        epochs=30,
        callbacks=[
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True
            )
        ]
    )

    return model, history, loader


def shap_analysis_kernel(model, loader, class_names):
    background_samples, _ = loader.get_sample_data(50)
    test_samples, test_labels = loader.get_sample_data(10)

    # Create explainer
    def predict_fn(imgs):
        return model.predict(imgs, batch_size=2, verbose=0)

    explainer = shap.KernelExplainer(
        predict_fn,
        shap.sample(background_samples, 10)
    )

    # calculate SHAP
    shap_values = []
    for i in range(0, 10, 2):
        batch = test_samples[i:i + 2]
        shap_values_batch = explainer.shap_values(batch, nsamples=50)
        shap_values.append(shap_values_batch)

    # combine the result
    final_shap = [np.concatenate([s[i] for s in shap_values], axis=0) for i in range(len(shap_values[0]))]

    # Visable
    plt.figure(figsize=(20, 6 * shap_sample_size))
    shap.image_plot(
        final_shap,
        test_samples,
        true_labels=[class_names[l] for l in test_labels],
        class_names=class_names,
        show=False
    )
    plt.suptitle("SHAP Value Analysis", y=0.95, fontsize=14)
    plt.savefig('shap_analysis.png', bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    model, history, loader = train_model()

    # Save Model
    model.save('alzheimer_model_optimized(Final Version).h5')
    print("model saved successfully")