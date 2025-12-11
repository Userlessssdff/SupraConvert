# app.py - Ultra Image Convert Backend (Robust & Thread-Safe)

import os
import secrets
import json
import time
import logging
from logging.handlers import RotatingFileHandler
import threading
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor
import io
# Removed requests import as AI Upscale is removed
from flask import Flask, request, jsonify, send_file, url_for, render_template
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv
from PIL import Image, ImageOps, ImageEnhance, ImageSequence
from PIL.Image import Resampling

# 0. Configuration & Initialization


load_dotenv()

# Logging Setup (Production Ready)
handler = RotatingFileHandler('app.log', maxBytes=100000, backupCount=3)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    handlers=[handler, logging.StreamHandler()]
)

# --- Thread Safety ---
JOB_LOCK = threading.Lock()
WORKER_POOL = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
JOB_QUEUE = {}

# --- File and Directory Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.getenv('TEMP_DIR', os.path.join(BASE_DIR, 'tmp', 'uploads'))
OUTPUT_FOLDER = os.getenv('OUTPUT_FOLDER', os.path.join(BASE_DIR, 'static', 'output'))

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Unified Cleanup Configuration ---
CLEANUP_INTERVAL_MINUTES = 30
CLEANUP_LOOP_SECONDS = CLEANUP_INTERVAL_MINUTES * 60

# --- Conversion Map (Features 16, 17, 18: New Formats) ---
CONVERSION_MAP = {
    'pillow_jpg': {'name': 'JPEG (.jpg)', 'ext': 'jpg', 'format': 'JPEG', 'supports_quality': True,
                   'mime': 'image/jpeg'},
    'pillow_png': {'name': 'PNG (.png)', 'ext': 'png', 'format': 'PNG', 'supports_quality': False, 'mime': 'image/png'},
    'pillow_webp': {'name': 'WebP (.webp)', 'ext': 'webp', 'format': 'WEBP', 'supports_quality': True,
                    'mime': 'image/webp'},
    'pillow_gif': {'name': 'GIF (.gif)', 'ext': 'gif', 'format': 'GIF', 'supports_quality': False, 'mime': 'image/gif'},
    'pillow_bmp': {'name': 'BMP (.bmp)', 'ext': 'bmp', 'format': 'BMP', 'supports_quality': False, 'mime': 'image/bmp'},
    'pillow_tiff': {'name': 'TIFF (.tiff)', 'ext': 'tiff', 'format': 'TIFF', 'supports_quality': True,
                    'mime': 'image/tiff'},
    'pillow_ico': {'name': 'ICO (Favicon)', 'ext': 'ico', 'format': 'ICO', 'supports_quality': False,
                   'mime': 'image/x-icon'},
    'pillow_avif': {'name': 'AVIF (.avif) - Lossy/Lossless', 'ext': 'avif', 'format': 'AVIF', 'supports_quality': True,
                    'mime': 'image/avif'},
    'pillow_jxl': {'name': 'JpegXL (.jxl) - Lossy/Lossless', 'ext': 'jxl', 'format': 'JXL', 'supports_quality': True,
                   'mime': 'image/jxl'},
    'pillow_heif': {'name': 'HEIF (.heif) - High Efficiency', 'ext': 'heif', 'format': 'HEIF', 'supports_quality': True,
                    'mime': 'image/heif'},
}

# 1. Flask App Factory & Extensions


app = Flask(__name__, static_url_path='/static', static_folder='static', template_folder='.')

app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "ultra-secret-image-key")
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_MB", 500)) * 1024 * 1024

CORS(app, resources={r"/api/*": {"origins": "*"}})

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["100 per minute"],
    storage_uri="memory://",
)


@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"msg": "Rate limit exceeded. Please slow down."}), 429


@app.errorhandler(413)
def too_large(e):
    limit_mb = int(app.config["MAX_CONTENT_LENGTH"] / (1024 * 1024))
    return jsonify({"msg": f"File size limit exceeded ({limit_mb}MB)"}), 413


@app.route('/')
def index():
    return render_template('index.html')


# --- Helper Functions (No changes) ---

def custom_secure_filename(filename):
    """Sanitizes filename safely."""
    filename = os.path.basename(filename).strip()
    filename = filename.replace(" ", "_")
    safe_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-')
    sanitized = "".join(c for c in filename if c in safe_chars).strip()
    if not sanitized: sanitized = "image"
    if len(sanitized) > 64:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:60] + ext
    return f"{secrets.token_urlsafe(6)}_{sanitized}"


def get_file_magic_mime(file_stream):
    """Safely attempts to detect MIME type."""
    file_stream.seek(0)
    header = file_stream.read(2048)
    file_stream.seek(0)
    try:
        # Note: Requires python-magic library in the environment
        from magic import from_buffer
        return from_buffer(header, mime=True)
    except ImportError:
        logging.warning("python-magic not installed, using fallback validation.")
        return 'application/octet-stream'
    except Exception as e:
        logging.error(f"Mime detection error: {e}")
        return 'application/octet-stream'


# 2. Image Processing Workers (Updated to remove upscale path)

def run_processing_job(job_id):
    """Routes the job to the correct worker function based on job type."""
    with JOB_LOCK:
        job = JOB_QUEUE.get(job_id)
        if not job: return

    mode = job.get('mode')

    if mode == 'convert':
        run_conversion_job(job_id)
    else:
        with JOB_LOCK:
            job['status'] = 'failed'
            job['log'] = "Invalid job mode. Only 'convert' is supported."


def run_conversion_job(job_id):
    """Worker thread with new advanced features, Opacity & Quality logic."""
    with JOB_LOCK:
        job = JOB_QUEUE.get(job_id)

    if not job: return

    with JOB_LOCK:
        job['status'] = 'running'
        job['progress'] = 10

    input_filepath = job['input_filepath']
    conversion_key = job['conversion_key']
    settings = job['settings']

    details = CONVERSION_MAP.get(conversion_key)
    if not details:
        with JOB_LOCK:
            job['status'] = 'failed'
            job['log'] = "Invalid conversion key."
        return

    target_format = details['format']
    output_ext = details['ext']

    base_name = os.path.splitext(os.path.basename(job['input_original_filename']))[0]
    output_filename = f"processed_{base_name}_{secrets.token_urlsafe(4)}.{output_ext}"
    output_filepath = os.path.join(OUTPUT_FOLDER, output_filename)

    with JOB_LOCK:
        job['output_filepath'] = output_filepath

    try:
        with JOB_LOCK:
            job['progress'] = 20
        img = Image.open(input_filepath)

        # Handle animations (seek 0 unless GIF/WEBP target)
        if getattr(img, "is_animated", False) and target_format not in ['GIF', 'WEBP']:
            img.seek(0)

        # Determine the PIL resampling filter based on user setting (Feature 22)
        resample_filter = Resampling.LANCZOS # High quality default
        req_resample = settings.get('resample_filter', 'lanczos')
        if req_resample == 'nearest':
            resample_filter = Resampling.NEAREST
        elif req_resample == 'bilinear':
            resample_filter = Resampling.BILINEAR
        elif req_resample == 'bicubic':
            resample_filter = Resampling.BICUBIC
        # LANCZOS (High quality) is the fallback/default


        # A. RESIZING (Done first to save processing time on pixels)

        req_width = settings.get('width')
        req_height = settings.get('height')
        maintain_ar = settings.get('maintain_ar')

        try:
            w_int = int(req_width) if req_width else 0
            h_int = int(req_height) if req_height else 0
        except ValueError:
            w_int, h_int = 0, 0

        if w_int > 0 or h_int > 0:
            current_w, current_h = img.size
            final_w = w_int if w_int > 0 else current_w
            final_h = h_int if h_int > 0 else current_h

            if maintain_ar and (w_int > 0 or h_int > 0):
                # Maintain aspect ratio by fitting/containing the image
                if w_int > 0 and h_int > 0:
                     # Fit inside the bounding box
                    img.thumbnail((w_int, h_int), resample=resample_filter)
                elif w_int > 0:
                    ratio = w_int / current_w
                    final_h = int(current_h * ratio)
                    img = img.resize((w_int, final_h), resample=resample_filter)
                elif h_int > 0:
                    ratio = h_int / current_h
                    final_w = int(current_w * ratio)
                    img = img.resize((final_w, h_int), resample=resample_filter)

            elif w_int > 0 and h_int > 0:
                # Force resize/stretch
                img = img.resize((w_int, h_int), resample=resample_filter)


        with JOB_LOCK:
            job['progress'] = 30

        # B. ROTATION AND FLIPPING (Features 1, 2)

        rotation = settings.get('rotation', '0')
        flip = settings.get('flip', 'none')

        if rotation != '0':
            # Rotate by degrees (must be 90, 180, 270)
            img = img.rotate(int(rotation), expand=True)

        if flip == 'horizontal':
            img = ImageOps.mirror(img)
        elif flip == 'vertical':
            img = ImageOps.flip(img)

        # C. COLOR ADJUSTMENTS (Features 3, 4, 5, 6)

        # Convert to RGB/RGBA before applying enhancements if not already
        if img.mode not in ('RGB', 'RGBA', 'L'):
             img = img.convert('RGB')

        # Grayscale (Feature 3)
        if settings.get('grayscale'):
            img = img.convert('L') # Convert to Grayscale
            if img.mode != 'RGB':
                 img = img.convert('RGB') # Convert back to RGB for subsequent enhancements/saving

        # Brightness (Feature 4)
        brightness_val = float(settings.get('brightness', 100)) / 100.0
        if brightness_val != 1.0:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness_val)

        # Contrast (Feature 5)
        contrast_val = float(settings.get('contrast', 100)) / 100.0
        if contrast_val != 1.0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast_val)

        # Sharpness (Feature 6)
        sharpness_val = float(settings.get('sharpness', 100)) / 100.0
        if sharpness_val != 1.0:
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(sharpness_val)

        with JOB_LOCK:
            job['progress'] = 40

        # D. OPACITY/TRANSPARENCY HANDLING (Original Opacity Logic)

        try:
            opacity_val = int(settings.get('opacity', 100))
        except (ValueError, TypeError):
            opacity_val = 100

        # Only apply if user requested < 100% opacity
        if 0 <= opacity_val < 100:
            # 1. Ensure we are in RGBA mode to manipulate alpha
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            # 2. Get the Alpha channel
            alpha = img.split()[3]
            # 3. Apply the factor (opacity / 100)
            factor = opacity_val / 100.0
            alpha = alpha.point(lambda p: int(p * factor))
            # 4. Put the modified alpha back
            img.putalpha(alpha)

        with JOB_LOCK:
            job['progress'] = 60

        # E. COLOR MODE & FORMAT HANDLING

        # If target doesn't support transparency (JPEG, BMP), we must composite
        if target_format in ['JPEG', 'BMP']:
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[3])
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
        elif target_format == 'ICO':
            img = ImageOps.contain(img, (256, 256))

        with JOB_LOCK:
            job['progress'] = 80

        # F. SAVING (Updated with new save options: Features 4, 5, 21, 16, 17, 18)

        save_params = {}

        if details['supports_quality']:
            quality_val = int(settings.get('quality', 95))
            save_params['quality'] = quality_val

        # Feature 5: JPEG Progressive Scan
        if target_format == 'JPEG' and settings.get('progressive_jpeg'):
            save_params['progressive'] = True

        # Feature 4: PNG Optimization
        if target_format == 'PNG' and settings.get('optimize_png'):
            save_params['optimize'] = True

        # Feature 21: WebP Lossless/Lossy
        if target_format == 'WEBP':
            is_lossless = settings.get('webp_lossless', False)
            save_params['lossless'] = is_lossless
            if not is_lossless:
                 if 'quality' not in save_params:
                     save_params['quality'] = 95
            else:
                save_params.pop('quality', None)

        # Features 16, 17, 18: AVIF, JXL, HEIF Lossless/Lossy support
        if target_format in ['AVIF', 'JXL', 'HEIF']:
            is_lossless = settings.get('lossless_modern', False)
            save_params['lossless'] = is_lossless
            if not is_lossless:
                 if 'quality' not in save_params:
                     save_params['quality'] = 95
            else:
                save_params.pop('quality', None)

        # If saving animated GIF/WEBP, handle all frames
        if target_format in ['GIF', 'WEBP'] and getattr(img, "is_animated", False):
            frames = [frame.copy() for frame in ImageSequence.Iterator(img)]
            img.save(output_filepath, format=target_format, save_all=True, append_images=frames[1:], **save_params)
        else:
            img.save(output_filepath, format=target_format, **save_params)

        with JOB_LOCK:
            job['end_time'] = datetime.now(timezone.utc).isoformat()
            job['status'] = 'completed'
            job['progress'] = 100
            job['log'] = f"Conversion to {target_format} complete."

    except Exception as e:
        app.logger.error(f"Processing job {job_id} failed: {e}")
        with JOB_LOCK:
            job['status'] = 'failed'
            job['log'] = str(e)


# 3. API Routes (Fixed and Updated)

@app.route('/api/config', methods=['GET'])
def get_config():
    """Returns the conversion map to the frontend."""
    return jsonify({"conversions": CONVERSION_MAP}), 200

@app.route('/api/upload', methods=['POST'])
@limiter.limit("20 per hour")
def upload_file():
    """FIX: Handles file upload and queues the conversion job."""
    if 'file' not in request.files:
        return jsonify({"msg": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"msg": "No selected file"}), 400

    conversion_key = request.form.get('conversion_key')
    settings_json = request.form.get('settings')
    mode = request.form.get('mode', 'convert')

    if not conversion_key or not settings_json:
        return jsonify({"msg": "Missing conversion key or settings."}), 400

    try:
        settings = json.loads(settings_json)
    except json.JSONDecodeError:
        return jsonify({"msg": "Invalid JSON for settings."}), 400

    if mode != 'convert':
        return jsonify({"msg": "Unsupported operation mode."}), 400

    # 1. Server-Side Validation & MIME Check
    file_stream = io.BytesIO(file.read())
    file.seek(0)
    mime_type = get_file_magic_mime(file_stream)
    if not mime_type.startswith('image/'):
         return jsonify({"msg": f"Unsupported file type: {mime_type}. Must be an image."}), 415

    # 2. Save file
    original_filename = custom_secure_filename(file.filename)
    job_uuid = secrets.token_urlsafe(16)
    temp_filename = f"{job_uuid}_{original_filename}"
    temp_filepath = os.path.join(TEMP_DIR, temp_filename)
    file.save(temp_filepath)

    # 3. Create Job and Queue
    new_job = {
        'job_uuid': job_uuid,
        'mode': mode,
        'status': 'queued',
        'progress': 0,
        'log': 'Job received and queued.',
        'start_time': datetime.now(timezone.utc).isoformat(),
        'input_filepath': temp_filepath,
        'input_original_filename': original_filename,
        'conversion_key': conversion_key,
        'settings': settings,
    }

    with JOB_LOCK:
        JOB_QUEUE[job_uuid] = new_job

    # 4. Schedule processing in the background
    WORKER_POOL.submit(run_processing_job, job_uuid)

    return jsonify({"msg": "Upload successful", "job_uuid": job_uuid}), 200


@app.route('/api/status/<string:job_uuid>', methods=['GET'])
def get_job_status(job_uuid):
    with JOB_LOCK:
        job = JOB_QUEUE.get(job_uuid)
    if not job: return jsonify({"msg": "Not found"}), 404
    job_data = job.copy()

    # Store the input file name for download naming in the frontend
    job_data['input_original_filename'] = job.get('input_original_filename')

    if job_data['status'] == 'completed' and job_data.get('output_filepath'):
        job_data['download_url'] = url_for('download_file', job_uuid=job_uuid, _external=True)

        # Remove output_filepath from the response payload
        job_data.pop('output_filepath', None)

        # Only 'convert' mode remains. Remove input file immediately to save space.
        try:
            if os.path.exists(job_data['input_filepath']):
                os.remove(job_data['input_filepath'])
        except Exception as e:
            app.logger.warning(f"Failed to remove input file for job {job_uuid}: {e}")

        # Remove input_filepath from the response payload
        if 'input_filepath' in job_data:
             job_data.pop('input_filepath', None)

    return jsonify(job_data), 200


@app.route('/api/download/<string:job_uuid>', methods=['GET'])
def download_file(job_uuid):
    with JOB_LOCK:
        job = JOB_QUEUE.get(job_uuid)
    if not job or job['status'] != 'completed' or not job.get('output_filepath'):
        return jsonify({"msg": "Unavailable"}), 404
    output_filepath = job['output_filepath']
    details = CONVERSION_MAP.get(job['conversion_key'])
    output_filename = os.path.basename(output_filepath)

    try:
        return send_file(
            output_filepath,
            mimetype=details['mime'],
            as_attachment=True,
            download_name=output_filename
        )
    except Exception as e:
        app.logger.error(f"Error serving output file for {job_uuid}: {e}")
        return jsonify({"msg": "Server Error"}), 500


if __name__ == '__main__':
    # Start the cleanup scheduler if needed, not shown here for brevity
    # start_cleanup_scheduler()
    app.run(debug=os.getenv('FLASK_DEBUG', 'False') == 'True', host='0.0.0.0', port=5000)
