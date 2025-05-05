from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import Metric
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import io
import os
import logging
from PIL import Image

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Compatibility patch for numpy
np.object = object  # Must be before other imports

app = Flask(__name__)

class SEBlock(tf.keras.layers.Layer):
    def __init__(self, ratio=8, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        # Use the channels_last format uniformly
        self.channel = input_shape[-1]

        # All layers must be explicitly defined
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.reshape = tf.keras.layers.Reshape((1, 1, self.channel))
        self.fc1 = tf.keras.layers.Dense(
            self.channel // self.ratio,
            activation='relu',
            kernel_initializer='he_normal',
            use_bias=False
        )
        self.fc2 = tf.keras.layers.Dense(
            self.channel,
            activation='sigmoid',
            kernel_initializer='he_normal',
            use_bias=False
        )
        super(SEBlock, self).build(input_shape)

    def call(self, inputs):
        # 保持原始计算流程
        se_feature = self.global_avg_pool(inputs)
        se_feature = self.reshape(se_feature)
        se_feature = self.fc1(se_feature)
        se_feature = self.fc2(se_feature)
        return tf.keras.layers.Multiply()([inputs, se_feature])

    def get_config(self):
        config = super().get_config()
        config.update({'ratio': self.ratio})
        return config

# Custom metric class
class F1Score(Metric):
    def __init__(self, name='f1_score', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision(thresholds=threshold)
        self.recall = tf.keras.metrics.Recall(thresholds=threshold)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + 1e-6))


# Model configuration
model_path = os.path.abspath('../alzheimer_model_optimized(Final Version).h5')

try:
    model = load_model(
        model_path,
        custom_objects={
            'F1Score': F1Score,
            'SEBlock': SEBlock  # Add a custom layer
            # Add other custom objects if needed
        },
        compile=False
    )
except Exception as e:
    raise RuntimeError(f"Model loading failed: {str(e)}") from e

# Constants
IMAGE_SIZE = (128,128)
CLASS_MAPPING = {
    0: 'Non Demented',
    1: 'Mild Dementia',
    2: 'Moderate Dementia',
    3: 'Very mild Dementia'
}


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/Recognition', methods=['GET', 'POST'])
def recognition():
    if request.method == 'GET':
        # Render the HTML page where the user can upload an image
        return render_template('Recognition.html')

    elif request.method == 'POST':
        # Handle the image file uploaded via POST request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file:
            # Read and preprocess the image
            image = Image.open(file.stream)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = image.resize(IMAGE_SIZE)  # Use the corrected dimensions

            # Convert to the model input format
            img_array = img_to_array(image)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Predict and process the results
            prediction_probabilities = model.predict(img_array)

            # Processing of four classification results
            probabilities = {
                CLASS_MAPPING[i]: float(prediction_probabilities[0][i])
                for i in range(4)
            }
            predicted_class = CLASS_MAPPING[np.argmax(prediction_probabilities)]

            return jsonify({
                'prediction': predicted_class,
                'probabilities': probabilities
            })
        else:
            return jsonify({'error': 'Method not allowed'}), 405

if __name__ == '__main__':
    # Production configuration
    app.run(
        host='0.0.0.0',
        port=6006,
        debug=False,
        use_reloader=False
    )