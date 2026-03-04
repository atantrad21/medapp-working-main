import os
import io
import sys
import logging
import traceback
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['REQUEST_TIMEOUT'] = 300  # 5 minutes

# Google Drive file IDs for 800+ checkpoint models
FILE_IDS = {
    'f': '1dMvJtRBb32BnGI8xc5lJd0U-NbJh90fT',
    'g': '11VoWUJ5Iq30HgBfLyTF5mnczk7DLiOFN',
    'i': '1QIvFXO0LzDa6IH683OWXkedRAXpcDvk-',
    'j': '1tNSVLfubqvFv5ACR_8B8Dp47UnsC9-He'
}

# Global state
models = {}
models_loaded = False
model_load_error = None

# Import TensorFlow with error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    logger.info(f"✅ TensorFlow {tf.__version__} imported successfully")
    
    # Configure TensorFlow for production
    tf.config.set_soft_device_placement(True)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"Found {len(gpus)} GPU(s)")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        logger.info("No GPU found, using CPU")
        
except Exception as e:
    logger.error(f"❌ Failed to import TensorFlow: {str(e)}")
    logger.error(traceback.format_exc())
    sys.exit(1)

# Custom Instance Normalization Layer
class InstanceNormalization(tf.keras.layers.Layer):
    """Instance Normalization Layer for CycleGAN"""
    
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
        super().build(input_shape)
    
    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset
    
    def get_config(self):
        config = super().get_config()
        config.update({'epsilon': self.epsilon})
        return config

# Model Architecture Functions
def downsample(filters, size, apply_norm=True):
    """Downsampling block for U-Net generator"""
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
    """Upsampling block for U-Net generator"""
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
    """
    U-Net Generator Architecture
    Input: 64x64x1 grayscale image
    Output: 64x64x1 grayscale image
    """
    inputs = tf.keras.layers.Input(shape=[64, 64, 1], name='input_image')
    
    # Downsampling stack
    down_stack = [
        downsample(128, 4, apply_norm=False),  # (bs, 32, 32, 128)
        downsample(256, 4),                     # (bs, 16, 16, 256)
        downsample(256, 4),                     # (bs, 8, 8, 256)
        downsample(256, 4),                     # (bs, 4, 4, 256)
        downsample(256, 4),                     # (bs, 2, 2, 256)
        downsample(256, 4),                     # (bs, 1, 1, 256)
    ]
    
    # Upsampling stack
    up_stack = [
        upsample(256, 4, apply_dropout=True),   # (bs, 2, 2, 512)
        upsample(256, 4, apply_dropout=True),   # (bs, 4, 4, 512)
        upsample(256, 4),                       # (bs, 8, 8, 512)
        upsample(256, 4),                       # (bs, 16, 16, 512)
        upsample(128, 4),                       # (bs, 32, 32, 256)
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
    
    # Final layer
    last = tf.keras.layers.Conv2DTranspose(
        1, 4, strides=2, padding='same',
        kernel_initializer=tf.random_normal_initializer(0., 0.02),
        use_bias=True,
        activation='tanh',
        name='output_layer'
    )
    
    x = last(x)
    
    return keras.Model(inputs=inputs, outputs=x, name='unet_generator')

def download_and_load_models():
    """
    Download and load all 4 generator models from Google Drive
    This function runs on startup and loads 800+ checkpoint weights
    """
    global models, models_loaded, model_load_error
    
    logger.info("=" * 80)
    logger.info("I-TRANSLATION v5.0 ROBUST - PRODUCTION DEPLOYMENT")
    logger.info("=" * 80)
    logger.info(f"Startup Time: {datetime.now().isoformat()}")
    logger.info(f"Python Version: {sys.version}")
    logger.info(f"TensorFlow Version: {tf.__version__}")
    logger.info(f"NumPy Version: {np.__version__}")
    logger.info("=" * 80)
    
    try:
        # Import gdown for Google Drive downloads
        try:
            import gdown
            logger.info("✅ gdown library imported")
        except ImportError:
            logger.error("❌ gdown not installed. Installing...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            import gdown
            logger.info("✅ gdown installed successfully")
        
        logger.info("\n📥 DOWNLOADING 800+ CHECKPOINT MODELS FROM GOOGLE DRIVE")
        logger.info(f"Total Models: {len(FILE_IDS)}")
        
        for idx, (name, file_id) in enumerate(FILE_IDS.items(), 1):
            logger.info(f"\n[{idx}/{len(FILE_IDS)}] Processing Generator {name.upper()}")
            logger.info("-" * 60)
            
            output_path = f'/tmp/generator_{name}_800ckpt.h5'
            url = f'https://drive.google.com/uc?id={file_id}'
            
            # Download with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logger.info(f"  Downloading (attempt {attempt + 1}/{max_retries})...")
                    gdown.download(url, output_path, quiet=False)
                    
                    # Verify download
                    if os.path.exists(output_path):
                        file_size = os.path.getsize(output_path) / (1024 * 1024)
                        logger.info(f"  ✅ Downloaded: {file_size:.2f} MB")
                        break
                    else:
                        raise Exception("File not found after download")
                        
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"  ⚠️ Download failed: {str(e)}. Retrying...")
                    else:
                        raise Exception(f"Failed to download after {max_retries} attempts: {str(e)}")
            
            # Build model architecture
            logger.info(f"  Building U-Net architecture...")
            model = unet_generator()
            logger.info(f"  ✅ Architecture created")
            
            # Initialize model with dummy input
            logger.info(f"  Initializing layers...")
            dummy_input = tf.zeros((1, 64, 64, 1))
            _ = model(dummy_input, training=False)
            logger.info(f"  ✅ Layers initialized")
            
            # Load weights
            logger.info(f"  Loading 800+ checkpoint weights...")
            model.load_weights(output_path, by_name=True, skip_mismatch=True)
            logger.info(f"  ✅ Weights loaded successfully")
            
            # Store model
            models[name] = model
            logger.info(f"  ✅ Generator {name.upper()} READY FOR INFERENCE")
            
            # Cleanup downloaded file to save space
            try:
                os.remove(output_path)
                logger.info(f"  🗑️ Cleaned up downloaded file")
            except:
                pass
        
        models_loaded = True
        logger.info("\n" + "=" * 80)
        logger.info("🎉 ALL 4 GENERATORS LOADED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Loaded Models: {list(models.keys())}")
        logger.info(f"Total Generators: {len(models)}")
        logger.info(f"Status: READY FOR PRODUCTION")
        logger.info("=" * 80 + "\n")
        
    except Exception as e:
        model_load_error = str(e)
        models_loaded = False
        logger.error("\n" + "=" * 80)
        logger.error("❌ MODEL LOADING FAILED")
        logger.error("=" * 80)
        logger.error(f"Error: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        logger.error("=" * 80 + "\n")

def preprocess_image(image_bytes):
    """
    Preprocess uploaded image for model inference
    - Convert to grayscale
    - Resize to 64x64
    - Normalize to [-1, 1]
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert('L')  # Convert to grayscale
        img = img.resize((64, 64), Image.LANCZOS)  # Resize
        img_array = np.array(img)
        img_array = (img_array / 127.5) - 1.0  # Normalize to [-1, 1]
        img_tensor = img_array.reshape(1, 64, 64, 1).astype(np.float32)
        return img_tensor
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        raise

def postprocess_image(tensor):
    """
    Postprocess model output to PNG image
    - Denormalize from [-1, 1] to [0, 255]
    - Convert to PIL Image
    - Encode as PNG bytes
    """
    try:
        img_array = tensor[0]
        img_array = ((img_array + 1.0) * 127.5).astype(np.uint8)  # Denormalize
        img_array = img_array[:, :, 0]
        img = Image.fromarray(img_array, mode='L')
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return img_bytes
    except Exception as e:
        logger.error(f"Postprocessing error: {str(e)}")
        raise

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint
    Returns model loading status and system information
    """
    try:
        response = {
            'status': 'healthy' if models_loaded else 'loading',
            'timestamp': datetime.now().isoformat(),
            'version': '5.0.0-ROBUST',
            'architecture': 'U-Net 64x64 grayscale',
            'checkpoints': '800+',
            'models_loaded': models_loaded,
            'generators': list(models.keys()) if models_loaded else [],
            'tensorflow_version': tf.__version__,
            'error': model_load_error if model_load_error else None
        }
        
        status_code = 200 if models_loaded else 503
        return jsonify(response), status_code
        
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/convert', methods=['POST'])
def convert_image():
    """
    Convert uploaded image using all 4 generators
    Returns 4 different converted versions (base64 encoded)
    """
    request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    logger.info(f"\n{'=' * 80}")
    logger.info(f"NEW CONVERSION REQUEST [{request_id}]")
    logger.info(f"{'=' * 80}")
    
    try:
        # Check if models are loaded
        if not models_loaded:
            error_msg = "Models not loaded yet. Please wait for initialization to complete."
            if model_load_error:
                error_msg += f" Error: {model_load_error}"
            logger.warning(f"[{request_id}] {error_msg}")
            return jsonify({'error': error_msg}), 503
        
        # Validate request
        if 'image' not in request.files:
            logger.warning(f"[{request_id}] No image in request")
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            logger.warning(f"[{request_id}] Empty filename")
            return jsonify({'error': 'Empty filename'}), 400
        
        logger.info(f"[{request_id}] Filename: {image_file.filename}")
        
        # Read and preprocess image
        logger.info(f"[{request_id}] Reading image bytes...")
        image_bytes = image_file.read()
        logger.info(f"[{request_id}] Image size: {len(image_bytes)} bytes")
        
        logger.info(f"[{request_id}] Preprocessing image...")
        input_tensor = preprocess_image(image_bytes)
        logger.info(f"[{request_id}] ✅ Preprocessed to shape: {input_tensor.shape}")
        
        # Run inference with all 4 generators
        results = {}
        for name, model in models.items():
            logger.info(f"[{request_id}] Running Generator {name.upper()}...")
            
            try:
                # Run inference
                output_tensor = model(input_tensor, training=False)
                logger.info(f"[{request_id}]   Output shape: {output_tensor.shape}")
                
                # Postprocess
                output_bytes = postprocess_image(output_tensor.numpy())
                
                # Encode to base64
                import base64
                output_b64 = base64.b64encode(output_bytes.getvalue()).decode('utf-8')
                results[f'generator_{name}'] = output_b64
                
                logger.info(f"[{request_id}]   ✅ Generator {name.upper()} completed")
                
            except Exception as e:
                logger.error(f"[{request_id}]   ❌ Generator {name.upper()} failed: {str(e)}")
                logger.error(traceback.format_exc())
                # Continue with other generators even if one fails
                results[f'generator_{name}'] = None
        
        # Check if at least one generator succeeded
        successful_results = {k: v for k, v in results.items() if v is not None}
        if not successful_results:
            raise Exception("All generators failed to process the image")
        
        logger.info(f"[{request_id}] ✅ Conversion completed successfully")
        logger.info(f"[{request_id}] Successful generators: {len(successful_results)}/{len(models)}")
        logger.info(f"{'=' * 80}\n")
        
        return jsonify({
            'success': True,
            'request_id': request_id,
            'results': results,
            'successful_count': len(successful_results),
            'total_count': len(models)
        }), 200
        
    except Exception as e:
        logger.error(f"[{request_id}] ❌ CONVERSION FAILED")
        logger.error(f"[{request_id}] Error: {str(e)}")
        logger.error(f"[{request_id}] Traceback:\n{traceback.format_exc()}")
        logger.info(f"{'=' * 80}\n")
        
        return jsonify({
            'error': str(e),
            'request_id': request_id,
            'traceback': traceback.format_exc() if app.debug else None
        }), 500

@app.route('/batch-convert', methods=['POST'])
def batch_convert():
    """
    Convert multiple images in batch
    Accepts up to 4 images
    """
    request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    logger.info(f"\n{'=' * 80}")
    logger.info(f"NEW BATCH CONVERSION REQUEST [{request_id}]")
    logger.info(f"{'=' * 80}")
    
    try:
        # Check if models are loaded
        if not models_loaded:
            error_msg = "Models not loaded yet. Please wait for initialization to complete."
            if model_load_error:
                error_msg += f" Error: {model_load_error}"
            logger.warning(f"[{request_id}] {error_msg}")
            return jsonify({'error': error_msg}), 503
        
        # Get all uploaded files
        files = request.files.getlist('images')
        if not files:
            logger.warning(f"[{request_id}] No images in request")
            return jsonify({'error': 'No images provided'}), 400
        
        if len(files) > 4:
            logger.warning(f"[{request_id}] Too many images: {len(files)}")
            return jsonify({'error': 'Maximum 4 images allowed'}), 400
        
        logger.info(f"[{request_id}] Processing {len(files)} images")
        
        batch_results = []
        
        for idx, image_file in enumerate(files, 1):
            logger.info(f"[{request_id}] Image {idx}/{len(files)}: {image_file.filename}")
            
            try:
                # Read and preprocess
                image_bytes = image_file.read()
                input_tensor = preprocess_image(image_bytes)
                
                # Run inference
                results = {}
                for name, model in models.items():
                    output_tensor = model(input_tensor, training=False)
                    output_bytes = postprocess_image(output_tensor.numpy())
                    
                    import base64
                    output_b64 = base64.b64encode(output_bytes.getvalue()).decode('utf-8')
                    results[f'generator_{name}'] = output_b64
                
                batch_results.append({
                    'filename': image_file.filename,
                    'success': True,
                    'results': results
                })
                
                logger.info(f"[{request_id}]   ✅ Image {idx} completed")
                
            except Exception as e:
                logger.error(f"[{request_id}]   ❌ Image {idx} failed: {str(e)}")
                batch_results.append({
                    'filename': image_file.filename,
                    'success': False,
                    'error': str(e)
                })
        
        successful_count = sum(1 for r in batch_results if r.get('success'))
        logger.info(f"[{request_id}] ✅ Batch conversion completed")
        logger.info(f"[{request_id}] Success rate: {successful_count}/{len(files)}")
        logger.info(f"{'=' * 80}\n")
        
        return jsonify({
            'success': True,
            'request_id': request_id,
            'total_images': len(files),
            'successful_images': successful_count,
            'results': batch_results
        }), 200
        
    except Exception as e:
        logger.error(f"[{request_id}] ❌ BATCH CONVERSION FAILED")
        logger.error(f"[{request_id}] Error: {str(e)}")
        logger.error(f"[{request_id}] Traceback:\n{traceback.format_exc()}")
        logger.info(f"{'=' * 80}\n")
        
        return jsonify({
            'error': str(e),
            'request_id': request_id
        }), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file size limit exceeded"""
    return jsonify({
        'error': 'File too large',
        'max_size': '16 MB'
    }), 413

@app.errorhandler(500)
def internal_server_error(error):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        'error': 'Internal server error',
        'message': str(error)
    }), 500

# ============================================================================
# STARTUP
# ============================================================================

# Load models on startup
logger.info("Starting model loading process...")
download_and_load_models()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"\n{'=' * 80}")
    logger.info(f"STARTING FLASK SERVER ON PORT {port}")
    logger.info(f"{'=' * 80}\n")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
