"""
I-TRANSLATION v4.7.7-ROBUST
Medical Image Conversion Backend with Robust Google Drive Downloads
Fixes: Google Drive download failures with retry logic and session management
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
import logging
import requests
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 * 1024  # 10GB

# Model storage
GENERATORS = {}

# Google Drive file IDs for 652 checkpoint weights
MODEL_FILES = {
    'F': '1O1hQSOoizPt5fJyVuEfxRpq0LibmaGeM',
    'G': '1nQnBaEyjQyTp3LJ6DF9tfaXrZxIHkROQ',
    'I': '1QIvFXO0LzDa6IH683OWXkedRAXpcDvk-',
    'J': '1-Quu4cDJhTpH7RDj-HZ-6c4VsQl1mc6j'
}

# Custom InstanceNormalization layer
class InstanceNormalization(layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=(input_shape[-1],),
            initializer='ones',
            trainable=True
        )
        self.offset = self.add_weight(
            name='offset',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        normalized = (inputs - mean) / tf.sqrt(variance + self.epsilon)
        return self.scale * normalized + self.offset

    def get_config(self):
        config = super(InstanceNormalization, self).get_config()
        config.update({'epsilon': self.epsilon})
        return config

def download_from_google_drive(file_id, destination, max_retries=3):
    """
    Robust Google Drive download with retry logic and virus scan handling.
    Uses requests library with session management for better reliability.
    """
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    session = requests.Session()
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Download attempt {attempt + 1}/{max_retries} for file ID: {file_id}")
            
            # First request to get confirmation token
            response = session.get(url, stream=True)
            
            # Check for virus scan warning
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={value}"
                    response = session.get(url, stream=True)
                    break
            
            # Download the file
            if response.status_code == 200:
                with open(destination, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # Verify file was downloaded
                if os.path.exists(destination):
                    file_size = os.path.getsize(destination)
                    if file_size > 1024 * 1024:  # At least 1 MB
                        logger.info(f"✅ Downloaded successfully! Size: {file_size / (1024*1024):.2f} MB")
                        return True
                    else:
                        logger.warning(f"⚠️ Downloaded file too small: {file_size} bytes")
                        os.remove(destination)
                else:
                    logger.error("❌ File not found after download")
            else:
                logger.error(f"❌ HTTP {response.status_code}: {response.reason}")
            
            # Wait before retry
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.info(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                
        except Exception as e:
            logger.error(f"❌ Download error: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)
    
    return False

def downsample(filters, size, name_prefix, apply_norm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = keras.Sequential(name=f"{name_prefix}_downsample")
    result.add(layers.Conv2D(
        filters, size, strides=2, padding='same',
        kernel_initializer=initializer, use_bias=False,
        name=f"{name_prefix}_conv"
    ))
    if apply_norm:
        result.add(InstanceNormalization(name=f"{name_prefix}_norm"))
    result.add(layers.LeakyReLU(name=f"{name_prefix}_leaky"))
    return result

def upsample(filters, size, name_prefix, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = keras.Sequential(name=f"{name_prefix}_upsample")
    result.add(layers.Conv2DTranspose(
        filters, size, strides=2, padding='same',
        kernel_initializer=initializer, use_bias=False,
        name=f"{name_prefix}_convT"
    ))
    result.add(InstanceNormalization(name=f"{name_prefix}_norm"))
    if apply_dropout:
        result.add(layers.Dropout(0.5, name=f"{name_prefix}_dropout"))
    result.add(layers.ReLU(name=f"{name_prefix}_relu"))
    return result

def unet_generator(name='generator'):
    inputs = layers.Input(shape=[256, 256, 3], name=f"{name}_input")
    
    down_stack = [
        downsample(64, 4, f"{name}_down1", apply_norm=False),
        downsample(128, 4, f"{name}_down2"),
        downsample(256, 4, f"{name}_down3"),
        downsample(512, 4, f"{name}_down4"),
        downsample(512, 4, f"{name}_down5"),
        downsample(512, 4, f"{name}_down6"),
        downsample(512, 4, f"{name}_down7"),
        downsample(512, 4, f"{name}_down8"),
    ]
    
    up_stack = [
        upsample(512, 4, f"{name}_up1", apply_dropout=True),
        upsample(512, 4, f"{name}_up2", apply_dropout=True),
        upsample(512, 4, f"{name}_up3", apply_dropout=True),
        upsample(512, 4, f"{name}_up4"),
        upsample(256, 4, f"{name}_up5"),
        upsample(128, 4, f"{name}_up6"),
        upsample(64, 4, f"{name}_up7"),
    ]
    
    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(
        3, 4, strides=2, padding='same',
        kernel_initializer=initializer,
        activation='tanh',
        name=f"{name}_output"
    )
    
    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    
    skips = reversed(skips[:-1])
    
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate(name=f"{up.name}_concat")([x, skip])
    
    x = last(x)
    
    return keras.Model(inputs=inputs, outputs=x, name=name)

def load_models():
    """Download weights from Google Drive and load into generators"""
    logger.info("=" * 60)
    logger.info("STARTING MODEL LOADING PROCESS")
    logger.info("=" * 60)
    
    loaded_count = 0
    
    for gen_name, file_id in MODEL_FILES.items():
        try:
            logger.info(f"\n[{gen_name}] Starting download and load process...")
            
            # Download weights
            weights_path = f'/tmp/generator_{gen_name.lower()}.h5'
            success = download_from_google_drive(file_id, weights_path)
            
            if not success:
                logger.error(f"[{gen_name}] ❌ Download failed after all retries")
                continue
            
            # Build model architecture
            logger.info(f"[{gen_name}] Building U-Net architecture...")
            model = unet_generator(name=f'generator_{gen_name}')
            
            # Initialize with dummy input
            logger.info(f"[{gen_name}] Initializing layers...")
            dummy_input = tf.random.normal([1, 256, 256, 3])
            _ = model(dummy_input)
            
            # Load weights
            logger.info(f"[{gen_name}] Loading weights from {weights_path}...")
            model.load_weights(weights_path, by_name=True, skip_mismatch=False)
            
            # Store model
            GENERATORS[gen_name] = model
            loaded_count += 1
            
            logger.info(f"[{gen_name}] ✅ Weights loaded successfully!")
            
            # Clean up
            if os.path.exists(weights_path):
                os.remove(weights_path)
                logger.info(f"[{gen_name}] Cleaned up weight file")
                
        except Exception as e:
            logger.error(f"[{gen_name}] ❌ Error loading model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    logger.info("\n" + "=" * 60)
    if loaded_count == 4:
        logger.info(f"✅ {loaded_count}/4 MODELS LOADED SUCCESSFULLY")
        logger.info("✅ APPLICATION READY TO SERVE REQUESTS")
    else:
        logger.warning(f"⚠️ {loaded_count}/4 MODELS LOADED")
        logger.warning("⚠️ SOME MODELS FAILED TO LOAD")
    logger.info("=" * 60)

def preprocess_image(image_bytes):
    """Convert uploaded image to model input format"""
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert('RGB')
    img = img.resize((256, 256))
    img_array = np.array(img)
    img_array = (img_array / 127.5) - 1.0
    return np.expand_dims(img_array, 0)

def postprocess_image(prediction):
    """Convert model output to displayable image"""
    img_array = (prediction[0] + 1.0) * 127.5
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'models_loaded': len(GENERATORS) == 4,
        'loaded_generators': list(GENERATORS.keys()),
        'version': '4.7.7-ROBUST'
    })

@app.route('/convert', methods=['POST'])
def convert():
    """Convert uploaded images using all loaded generators"""
    if len(GENERATORS) == 0:
        return jsonify({'error': 'No models loaded'}), 503
    
    try:
        results = {}
        
        for i in range(1, 5):
            image_key = f'image{i}'
            if image_key in request.files:
                file = request.files[image_key]
                if file.filename:
                    # Preprocess
                    img_bytes = file.read()
                    input_tensor = preprocess_image(img_bytes)
                    
                    # Convert with all generators
                    conversions = {}
                    for gen_name, model in GENERATORS.items():
                        prediction = model(input_tensor, training=False)
                        output_img = postprocess_image(prediction.numpy())
                        
                        # Convert to bytes
                        img_io = io.BytesIO()
                        output_img.save(img_io, 'PNG')
                        img_io.seek(0)
                        
                        conversions[gen_name] = img_io.getvalue().hex()
                    
                    results[image_key] = conversions
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Conversion error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Load models on startup
load_models()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
