import os
import io
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

FILE_IDS = {
    'f': '1dMvJtRBb32BnGI8xc5lJd0U-NbJh90fT',
    'g': '11VoWUJ5Iq30HgBfLyTF5mnczk7DLiOFN',
    'i': '1QIvFXO0LzDa6IH683OWXkedRAXpcDvk-',
    'j': '1tNSVLfubqvFv5ACR_8B8Dp47UnsC9-He'
}

models = {}
models_loaded = False

import tensorflow as tf
from tensorflow import keras
import h5py

logger.info("TensorFlow imported")
logger.info(f"TensorFlow version: {tf.__version__}")

class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
    
    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True
        )
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True
        )
    
    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset
    
    def get_config(self):
        config = super().get_config()
        config.update({'epsilon': self.epsilon})
        return config

def downsample(filters, size, apply_norm=True):
    result = keras.Sequential()
    result.add(tf.keras.layers.Conv2D(
        filters, size, strides=2, padding='same',
        kernel_initializer=tf.random_normal_initializer(0., 0.02),
        use_bias=False
    ))
    if apply_norm:
        result.add(InstanceNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    result = keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(
        filters, size, strides=2, padding='same',
        kernel_initializer=tf.random_normal_initializer(0., 0.02),
        use_bias=False
    ))
    result.add(InstanceNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result

def unet_generator():
    inputs = tf.keras.layers.Input(shape=[64, 64, 1])
    
    down_stack = [
        downsample(128, 4, apply_norm=False),
        downsample(256, 4),
        downsample(256, 4),
        downsample(256, 4),
        downsample(256, 4),
        downsample(256, 4),
    ]
    
    up_stack = [
        upsample(256, 4, apply_dropout=True),
        upsample(256, 4, apply_dropout=True),
        upsample(256, 4),
        upsample(256, 4),
        upsample(128, 4),
    ]
    
    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    
    skips = reversed(skips[:-1])
    
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    
    last = tf.keras.layers.Conv2DTranspose(
        1, 4, strides=2, padding='same',
        kernel_initializer=tf.random_normal_initializer(0., 0.02),
        use_bias=True,
        activation='tanh'
    )
    
    x = last(x)
    
    return keras.Model(inputs=inputs, outputs=x)

def load_weights_manually(model, h5_path):
    """Load weights from H5 file by manually reading weight arrays"""
    logger.info(f"Reading weight arrays from {h5_path}...")
    
    with h5py.File(h5_path, 'r') as f:
        # Extract all weight arrays
        weight_arrays = []
        
        # Try different H5 structures
        if 'model_weights' in f.keys():
            # Standard Keras format
            logger.info("Found model_weights group")
            for layer_name in f['model_weights'].keys():
                layer_group = f['model_weights'][layer_name]
                for weight_name in layer_group.keys():
                    weight_arrays.append(np.array(layer_group[weight_name]))
        else:
            # Raw weight arrays (numbered)
            logger.info("Searching for raw weight arrays...")
            for key in sorted(f.keys()):
                if isinstance(f[key], h5py.Dataset):
                    weight_arrays.append(np.array(f[key]))
        
        logger.info(f"Found {len(weight_arrays)} weight arrays")
        
        # Assign weights to model
        if len(weight_arrays) != len(model.weights):
            logger.warning(f"Weight count mismatch: file has {len(weight_arrays)}, model expects {len(model.weights)}")
        
        # Assign weights by index
        for i, (model_weight, file_weight) in enumerate(zip(model.weights, weight_arrays)):
            if model_weight.shape == file_weight.shape:
                model_weight.assign(file_weight)
                logger.info(f"Loaded weight {i}: {model_weight.shape}")
            else:
                logger.error(f"Shape mismatch at index {i}: model {model_weight.shape} vs file {file_weight.shape}")
                raise ValueError(f"Weight shape mismatch at index {i}")

def download_and_load_models():
    global models, models_loaded
    
    try:
        import gdown
        logger.info("DOWNLOADING 652 CHECKPOINT MODELS FROM GOOGLE DRIVE")
        
        for name, file_id in FILE_IDS.items():
            logger.info(f"Downloading Generator {name.upper()}...")
            output_path = f'/tmp/generator_{name}.h5'
            
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, output_path, quiet=False)
            
            logger.info(f"Building Generator {name.upper()} architecture...")
            model = unet_generator()
            
            logger.info(f"Initializing Generator {name.upper()} layers...")
            dummy_input = tf.zeros((1, 64, 64, 1))
            _ = model(dummy_input, training=False)
            
            logger.info(f"Model has {len(model.weights)} weight tensors")
            
            logger.info(f"Loading weights manually for Generator {name.upper()}...")
            load_weights_manually(model, output_path)
            
            models[name] = model
            logger.info(f"Generator {name.upper()} LOADED SUCCESSFULLY")
        
        models_loaded = True
        logger.info("ALL 4 GENERATORS LOADED - READY FOR INFERENCE")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        models_loaded = False

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert('L')
    img = img.resize((64, 64), Image.LANCZOS)
    img_array = np.array(img)
    img_array = (img_array / 127.5) - 1.0
    img_tensor = img_array.reshape(1, 64, 64, 1).astype(np.float32)
    return img_tensor

def postprocess_image(tensor):
    img_array = tensor[0]
    img_array = ((img_array + 1.0) * 127.5).astype(np.uint8)
    img_array = img_array[:, :, 0]
    img = Image.fromarray(img_array, mode='L')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'version': '5.0.0',
        'architecture': '64x64 grayscale',
        'checkpoints': 652,
        'models_loaded': models_loaded,
        'generators': list(models.keys()) if models_loaded else []
    }), 200

@app.route('/convert', methods=['POST'])
def convert_image():
    if not models_loaded:
        return jsonify({'error': 'Models not loaded yet'}), 503
    
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        image_bytes = image_file.read()
        input_tensor = preprocess_image(image_bytes)
        
        results = {}
        for name, model in models.items():
            output_tensor = model(input_tensor, training=False)
            output_bytes = postprocess_image(output_tensor.numpy())
            
            import base64
            output_b64 = base64.b64encode(output_bytes.getvalue()).decode('utf-8')
            results[f'generator_{name}'] = output_b64
        
        return jsonify({
            'success': True,
            'results': results
        }), 200
        
    except Exception as e:
        logger.error(f"Conversion error: {str(e)}")
        return jsonify({'error': str(e)}), 500

logger.info("I-TRANSLATION v5.0 - 652 CHECKPOINT PRODUCTION")
download_and_load_models()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
