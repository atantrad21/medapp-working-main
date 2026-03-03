"""
I-Translation v5.0.0 FINAL - 652 Checkpoint Production
QUAD-GAN Medical Imaging Translation System
64x64 Grayscale Architecture (13M parameters per generator)
"""

import os
import io
import logging
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask app initialization
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# ============================================================================
# REAL 652 CHECKPOINT FILE IDS (Fresh Training)
# ============================================================================
FILE_IDS = {
    'f': '1dMvJtRBb32BnGI8xc5lJd0U-NbJh90fT',
    'g': '11VoWUJ5Iq30HgBfLyTF5mnczk7DLiOFN',
    'i': '1QIvFXO0LzDa6IH683OWXkedRAXpcDvk-',
    'j': '1tNSVLfubqvFv5ACR_8B8Dp47UnsC9-He'
}

# Global models dictionary
models = {}
models_loaded = False

# ============================================================================
# TENSORFLOW AND MODEL ARCHITECTURE
# ============================================================================
import tensorflow as tf
from tensorflow import keras

logger.info("TensorFlow imported successfully")
logger.info(f"TensorFlow version: {tf.__version__}")

# Instance Normalization Layer
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

# Downsample block
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

# Upsample block
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

# U-Net Generator (64x64 grayscale, 13M parameters)
def unet_generator():
    inputs = tf.keras.layers.Input(shape=[64, 64, 1])
    
    # Downsampling
    down_stack = [
        downsample(128, 4, apply_norm=False),  # (bs, 32, 32, 128)
        downsample(256, 4),  # (bs, 16, 16, 256)
        downsample(256, 4),  # (bs, 8, 8, 256)
        downsample(256, 4),  # (bs, 4, 4, 256)
        downsample(256, 4),  # (bs, 2, 2, 256)
        downsample(256, 4),  # (bs, 1, 1, 256)
    ]
    
    # Upsampling
    up_stack = [
        upsample(256, 4, apply_dropout=True),  # (bs, 2, 2, 512)
        upsample(256, 4, apply_dropout=True),  # (bs, 4, 4, 512)
        upsample(256, 4),  # (bs, 8, 8, 512)
        upsample(256, 4),  # (bs, 16, 16, 512)
        upsample(128, 4),  # (bs, 32, 32, 256)
    ]
    
    # Downsampling through the model
    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    
    skips = reversed(skips[:-1])
    
    # Upsampling and establishing skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    
    # Final output layer
    last = tf.keras.layers.Conv2DTranspose(
        1, 4, strides=2, padding='same',
        kernel_initializer=tf.random_normal_initializer(0., 0.02),
        activation='tanh'
    )
    
    x = last(x)
    
    return keras.Model(inputs=inputs, outputs=x)

# ============================================================================
# MODEL LOADING
# ============================================================================
def download_and_load_models():
    """Download and load all 4 generator models"""
    global models, models_loaded
    
    try:
        import gdown
        logger.info("=" * 80)
        logger.info("DOWNLOADING 652 CHECKPOINT MODELS FROM GOOGLE DRIVE")
        logger.info("=" * 80)
        
        for name, file_id in FILE_IDS.items():
            logger.info(f"Downloading Generator {name.upper()}...")
            output_path = f'/tmp/generator_{name}.h5'
            
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, output_path, quiet=False)
            
            logger.info(f"Building Generator {name.upper()} architecture...")
            model = unet_generator()
            
            logger.info(f"Loading weights for Generator {name.upper()}...")
            model.load_weights(output_path)
            
            models[name] = model
            logger.info(f"✅ Generator {name.upper()} loaded successfully")
        
        models_loaded = True
        logger.info("=" * 80)
        logger.info("ALL 4 GENERATORS LOADED SUCCESSFULLY - READY FOR INFERENCE")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        logger.error(f"Full error: {repr(e)}")
        models_loaded = False

# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================
def preprocess_image(image_bytes):
    """Convert uploaded image to 64x64 grayscale tensor"""
    img = Image.open(io.BytesIO(image_bytes))
    
    # Convert to grayscale
    img = img.convert('L')
    
    # Resize to 64x64
    img = img.resize((64, 64), Image.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Normalize to [-1, 1]
    img_array = (img_array / 127.5) - 1.0
    
    # Add batch and channel dimensions
    img_tensor = img_array.reshape(1, 64, 64, 1).astype(np.float32)
    
    return img_tensor

# ============================================================================
# IMAGE POSTPROCESSING
# ============================================================================
def postprocess_image(tensor):
    """Convert model output tensor back to PNG image"""
    # Remove batch dimension
    img_array = tensor[0]
    
    # Denormalize from [-1, 1] to [0, 255]
    img_array = ((img_array + 1.0) * 127.5).astype(np.uint8)
    
    # Remove channel dimension
    img_array = img_array[:, :, 0]
    
    # Create PIL image
    img = Image.fromarray(img_array, mode='L')
    
    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    return img_bytes

# ============================================================================
# API ENDPOINTS
# ============================================================================
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
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
    """Single image conversion through all 4 generators"""
    if not models_loaded:
        return jsonify({'error': 'Models not loaded yet'}), 503
    
    try:
        # Get uploaded image
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        image_bytes = image_file.read()
        
        # Preprocess
        input_tensor = preprocess_image(image_bytes)
        
        # Run through all 4 generators
        results = {}
        for name, model in models.items():
            output_tensor = model(input_tensor, training=False)
            output_bytes = postprocess_image(output_tensor.numpy())
            
            # Convert to base64 for JSON response
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

@app.route('/convert-batch', methods=['POST'])
def convert_batch():
    """Batch image conversion"""
    if not models_loaded:
        return jsonify({'error': 'Models not loaded yet'}), 503
    
    try:
        files = request.files.getlist('images')
        if not files:
            return jsonify({'error': 'No images provided'}), 400
        
        all_results = []
        
        for idx, file in enumerate(files):
            image_bytes = file.read()
            input_tensor = preprocess_image(image_bytes)
            
            results = {}
            for name, model in models.items():
                output_tensor = model(input_tensor, training=False)
                output_bytes = postprocess_image(output_tensor.numpy())
                
                import base64
                output_b64 = base64.b64encode(output_bytes.getvalue()).decode('utf-8')
                results[f'generator_{name}'] = output_b64
            
            all_results.append({
                'image_index': idx,
                'results': results
            })
        
        return jsonify({
            'success': True,
            'batch_results': all_results
        }), 200
        
    except Exception as e:
        logger.error(f"Batch conversion error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ============================================================================
# STARTUP - Load models at module level for Gunicorn
# ============================================================================
logger.info("=" * 80)
logger.info("I-TRANSLATION v5.0.0 FINAL - 652 CHECKPOINT PRODUCTION")
logger.info("64x64 GRAYSCALE ARCHITECTURE")
logger.info("=" * 80)

# Load models immediately
download_and_load_models()

# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
