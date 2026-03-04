"""
I-Translation v5.0 - 800 Checkpoint Production Release
Medical Image Translation: CT <-> MRI
4 Independent Generators (F, G, I, J)
Architecture: 64x64 Grayscale Input/Output
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from PIL import Image
import io
import os
import time
import logging
import gdown

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Global variables
MODELS = {}
START_TIME = time.time()
MODELS_LOADED = False

# 800 Checkpoint Model File IDs - TEMPORARY UNTIL TRAINING COMPLETES
FILE_IDS = {
    'f': 'TEMP_WAITING_FOR_800CKPT_TRAINING_F',
    'g': 'TEMP_WAITING_FOR_800CKPT_TRAINING_G',
    'i': 'TEMP_WAITING_FOR_800CKPT_TRAINING_I',
    'j': 'TEMP_WAITING_FOR_800CKPT_TRAINING_J'
}

# ============================================================================
# CUSTOM LAYERS - Instance Normalization
# ============================================================================

class InstanceNormalization(layers.Layer):
    """Instance Normalization Layer"""
    
    def __init__(self, epsilon=1e-5, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=(input_shape[-1],),
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True
        )
        self.offset = self.add_weight(
            name='offset',
            shape=(input_shape[-1],),
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
        config.update({"epsilon": self.epsilon})
        return config

# ============================================================================
# MODEL ARCHITECTURE - Matching Your Training Code (64x64, 1 channel)
# ============================================================================

def downsample(filters, size, apply_norm=True, name=None):
    """Downsampling block - matches your training architecture"""
    initializer = tf.random_normal_initializer(0., 0.02)
    result = keras.Sequential(name=name)
    
    result.add(layers.Conv2D(
        filters, size, strides=2, padding='same',
        kernel_initializer=initializer, use_bias=False,
        name=f'{name}_conv' if name else None
    ))
    
    if apply_norm:
        result.add(InstanceNormalization(name=f'{name}_norm' if name else None))
    
    result.add(layers.LeakyReLU(name=f'{name}_relu' if name else None))
    return result

def upsample(filters, size, apply_dropout=False, name=None):
    """Upsampling block - matches your training architecture"""
    initializer = tf.random_normal_initializer(0., 0.02)
    result = keras.Sequential(name=name)
    
    result.add(layers.Conv2DTranspose(
        filters, size, strides=2, padding='same',
        kernel_initializer=initializer, use_bias=False,
        name=f'{name}_conv' if name else None
    ))
    
    result.add(InstanceNormalization(name=f'{name}_norm' if name else None))
    
    if apply_dropout:
        result.add(layers.Dropout(0.5, name=f'{name}_dropout' if name else None))
    
    result.add(layers.ReLU(name=f'{name}_relu' if name else None))
    return result

def unet_generator(name='generator'):
    """
    U-Net Generator - EXACT match to your training architecture
    Input: [64, 64, 1] (grayscale)
    Output: [64, 64, 1] (grayscale)
    33 weight arrays total
    """
    inputs = layers.Input(shape=[64, 64, 1], name=f'{name}_input')
    
    # Downsampling - matching your weight analysis
    down_stack = [
        downsample(128, 4, apply_norm=False, name=f'{name}_down1'),
        downsample(256, 4, name=f'{name}_down2'),
        downsample(256, 4, name=f'{name}_down3'),
        downsample(256, 4, name=f'{name}_down4'),
        downsample(256, 4, name=f'{name}_down5'),
        downsample(256, 4, name=f'{name}_down6'),
    ]
    
    # Upsampling - matching your weight analysis
    up_stack = [
        upsample(256, 4, apply_dropout=True, name=f'{name}_up1'),
        upsample(256, 4, apply_dropout=True, name=f'{name}_up2'),
        upsample(256, 4, apply_dropout=True, name=f'{name}_up3'),
        upsample(256, 4, name=f'{name}_up4'),
        upsample(128, 4, name=f'{name}_up5'),
    ]
    
    # Final output layer
    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(
        1, 4, strides=2, padding='same',
        kernel_initializer=initializer,
        activation='tanh',
        name=f'{name}_output'
    )
    
    x = inputs
    skips = []
    
    # Downsampling path
    for down in down_stack:
        x = down(x)
        skips.append(x)
    
    skips = reversed(skips[:-1])
    
    # Upsampling path with skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate(name=f'{name}_concat_{up.name}')([x, skip])
    
    x = last(x)
    
    return keras.Model(inputs=inputs, outputs=x, name=name)

# ============================================================================
# MODEL LOADING
# ============================================================================

def download_weights():
    """Download model weights from Google Drive using gdown"""
    logger.info("=" * 80)
    logger.info("DOWNLOADING 800 CHECKPOINT MODEL WEIGHTS FROM GOOGLE DRIVE")
    logger.info("=" * 80)
    
    # Check if we're using temporary IDs
    if 'TEMP' in FILE_IDS['f']:
        logger.warning("=" * 80)
        logger.warning("TEMPORARY FILE IDs DETECTED")
        logger.warning("Waiting for 800 checkpoint training to complete")
        logger.warning("Models will load once real Google Drive file IDs are provided")
        logger.warning("=" * 80)
        return False
    
    os.makedirs('weights', exist_ok=True)
    
    for gen_name, file_id in FILE_IDS.items():
        output_path = f'weights/generator_{gen_name}.h5'
        
        if os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(f"✓ Generator {gen_name.upper()}: Already exists ({size_mb:.1f} MB)")
            continue
        
        logger.info(f"⬇ Downloading Generator {gen_name.upper()} (800 ckpt)...")
        url = f'https://drive.google.com/uc?id={file_id}'
        
        try:
            gdown.download(url, output_path, quiet=False)
            
            if os.path.exists(output_path):
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                
                if size_mb < 1:
                    logger.error(f"✗ Generator {gen_name.upper()}: File too small ({size_mb:.1f} MB)")
                    os.remove(output_path)
                    return False
                
                logger.info(f"✓ Generator {gen_name.upper()}: Downloaded successfully ({size_mb:.1f} MB)")
            else:
                logger.error(f"✗ Generator {gen_name.upper()}: Download failed")
                return False
                
        except Exception as e:
            logger.error(f"✗ Generator {gen_name.upper()}: Error - {str(e)}")
            return False
    
    logger.info("=" * 80)
    logger.info("ALL 800 CHECKPOINT WEIGHTS DOWNLOADED SUCCESSFULLY")
    logger.info("=" * 80)
    return True

def load_models():
    """Load all 4 generator models with 800 checkpoint weights"""
    global MODELS, MODELS_LOADED
    
    logger.info("=" * 80)
    logger.info("LOADING 800 CHECKPOINT MODELS")
    logger.info("=" * 80)
    
    if not download_weights():
        logger.info("Models not loaded - waiting for training to complete")
        return False
    
    for gen_name in ['f', 'g', 'i', 'j']:
        try:
            logger.info(f"Building Generator {gen_name.upper()} architecture...")
            model = unet_generator(name=f'generator_{gen_name}')
            
            logger.info(f"Loading 800 checkpoint weights for Generator {gen_name.upper()}...")
            weight_path = f'weights/generator_{gen_name}.h5'
            model.load_weights(weight_path)
            
            MODELS[gen_name] = model
            logger.info(f"✓ Generator {gen_name.upper()}: Loaded successfully (800 ckpt)")
            
        except Exception as e:
            logger.error(f"✗ Generator {gen_name.upper()}: Failed - {str(e)}")
            return False
    
    MODELS_LOADED = True
    logger.info("=" * 80)
    logger.info("ALL 4 GENERATORS LOADED WITH 800 CHECKPOINT WEIGHTS")
    logger.info("=" * 80)
    return True

# ============================================================================
# IMAGE PROCESSING - Adapted for 64x64 grayscale
# ============================================================================

def preprocess_image(image_bytes):
    """Preprocess image for model input - 64x64 grayscale"""
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    img = img.resize((64, 64), Image.LANCZOS)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = (img_array / 127.5) - 1.0
    return np.expand_dims(img_array, axis=0)

def postprocess_image(prediction):
    """Convert model output to image"""
    img_array = (prediction<sup>0</sup> + 1.0) * 127.5
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    img_array = np.squeeze(img_array)
    return Image.fromarray(img_array, mode='L')

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    training_status = "WAITING FOR 800 CHECKPOINT TRAINING" if 'TEMP' in FILE_IDS['f'] else "800 checkpoints"
    
    return jsonify({
        'status': 'healthy',
        'version': 'v5.0-800-checkpoints-PRODUCTION',
        'tensorflow_version': tf.__version__,
        'models_loaded': MODELS_LOADED,
        'input_shape': '[64, 64, 1]',
        'output_shape': '[64, 64, 1]',
        'training': training_status,
        'uptime_seconds': time.time() - START_TIME
    })

@app.route('/convert/<conversion_type>/<generator>', methods=['POST'])
def convert_image(conversion_type, generator):
    """Convert image using specified generator"""
    
    if conversion_type not in ['ct-to-mri', 'mri-to-ct']:
        return jsonify({'error': 'Invalid conversion type'}), 400
    
    if generator not in ['f', 'g', 'i', 'j']:
        return jsonify({'error': 'Invalid generator'}), 400
    
    if not MODELS_LOADED:
        return jsonify({'error': 'Models not loaded - waiting for 800 checkpoint training to complete'}), 503
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        image_file = request.files['image']
        image_bytes = image_file.read()
        
        input_tensor = preprocess_image(image_bytes)
        model = MODELS[generator]
        prediction = model(input_tensor, training=False)
        result_image = postprocess_image(prediction.numpy())
        
        output = io.BytesIO()
        result_image.save(output, format='PNG')
        output.seek(0)
        
        return send_file(output, mimetype='image/png')
        
    except Exception as e:
        logger.error(f"Conversion error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ============================================================================
# STARTUP
# ============================================================================

logger.info("=" * 80)
logger.info("I-TRANSLATION v5.0 - 800 CHECKPOINT PRODUCTION (64x64 GRAYSCALE)")
logger.info("=" * 80)

if load_models():
    logger.info("✓ Server ready with 800 checkpoint models")
else:
    logger.info("⏳ Server ready - waiting for 800 checkpoint training to complete")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
