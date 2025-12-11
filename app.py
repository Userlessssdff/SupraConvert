# app.py - Ultra Image Convert Backend (Secured and Streamlined)

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
from PIL import Image, ImageOps, ImageEnhance, UnidentifiedImageError, ImageSequence  # Added necessary PIL imports
from PIL.Image import Resampling  # Import Resampling filter for high quality resize

from flask import Flask, request, jsonify, send_file, url_for, render_template
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv

# --- Attempt to import python-magic for better MIME security ---
try:
    import magic

    HAS_MAGIC = True
except ImportError:
    HAS_MAGIC = False
    logging.warning("python-magic not installed. Using fallback validation.")

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

# --- Conversion Map ---
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


# --- Helper Functions ---

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
    header = file_stream.read(4096)
    file_stream.seek(0)

    if HAS_MAGIC:
        try:
            return magic.from_buffer(header, mime=True)
        except Exception as e:
            logging.error(f"python-magic detection error: {e}")
            return 'application/octet-stream'
    else:
        # Fallback validation
        return 'application/octet-stream'


def cleanup_temp_file(filepath):
    """Deletes a file if it exists."""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception as e:
        app.logger.error(f"Failed to remove temp file {filepath}: {e}")


# 2. Image Processing Worker

def run_image_job(job_id):
    """Worker thread with Resizing, Opacity & Quality logic."""
    with JOB_LOCK:
        job = JOB_QUEUE.get(job_id)

    if not job: return

    input_filepath = job['input_filepath']

    try:
        with JOB_LOCK:
            job['status'] = 'running'
            job['progress'] = 10

        conversion_key = job['conversion_key']
        settings = job['settings']
        details = CONVERSION_MAP.get(conversion_key)

        target_format = details['format']
        output_ext = details['ext']

        base_name = os.path.splitext(os.path.basename(job['input_original_filename']))[0]
        output_filename = f"processed_{base_name}_{secrets.token_urlsafe(4)}.{output_ext}"
        output_filepath = os.path.join(OUTPUT_FOLDER, output_filename)

        with JOB_LOCK:
            job['output_filepath'] = output_filepath
            job['progress'] = 20

        # --- Image Opening and Structural Validation (Pre-Check) ---
        img = Image.open(input_filepath)
        img.load()

        if getattr(img, "is_animated", False) and target_format not in ['GIF', 'WEBP']:
            img.seek(0)

        resample_filter = Resampling.LANCZOS

        # A. RESIZING
        req_width = settings.get('width')
        req_height = settings.get('height')
        maintain_ar = settings.get('maintain_ar')

        w_int = int(req_width) if req_width and str(req_width).isdigit() else 0
        h_int = int(req_height) if req_height and str(req_height).isdigit() else 0

        if w_int > 0 or h_int > 0:
            current_w, current_h = img.size
            final_w = w_int if w_int > 0 else current_w
            final_h = h_int if h_int > 0 else current_h

            if maintain_ar:
                # Resize image to fit inside the bounding box, maintaining AR
                img = ImageOps.contain(img, (final_w, final_h), method=resample_filter)
            else:
                # Force resize/stretch
                img = img.resize((final_w, final_h), resample=resample_filter)

        with JOB_LOCK:
            job['progress'] = 40

        # B. OPACITY ADJUSTMENT
        opacity_val = int(settings.get('opacity', 100))
        if 0 <= opacity_val < 100:
            if img.mode != 'RGBA':
                img = img.convert('RGBA')

            alpha = img.split()[3]
            factor = opacity_val / 100.0
            alpha = alpha.point(lambda p: int(p * factor))
            img.putalpha(alpha)

        with JOB_LOCK:
            job['progress'] = 60

        # C. COLOR MODE & FORMAT HANDLING

        if target_format in ['JPEG', 'BMP']:
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode != 'RGBA': img = img.convert('RGBA')
                background.paste(img, mask=img.split()[3])
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')

        elif target_format == 'ICO':
            img = ImageOps.contain(img, (256, 256))

        with JOB_LOCK:
            job['progress'] = 80

        # D. SAVING & QUALITY
        save_kwargs = {}
        if details['supports_quality']:
            q = int(settings.get('quality', 90))
            save_kwargs['quality'] = max(1, min(100, q))

        if target_format == 'GIF' and getattr(img, "is_animated", False):
            save_kwargs['optimize'] = True
            frames = [frame.copy() for frame in ImageSequence.Iterator(img)]
            save_kwargs['save_all'] = True
            save_kwargs['append_images'] = frames[1:]

        img.save(output_filepath, format=target_format, **save_kwargs)
        img.close()

        # Final Verification
        if os.path.exists(output_filepath) and os.path.getsize(output_filepath) > 0:
            with JOB_LOCK:
                job['status'] = 'completed'
                job['progress'] = 100
                job['end_time'] = datetime.now(timezone.utc).isoformat()
                job['log'] = f"Conversion to {target_format} successful."
        else:
            raise Exception("File save failed (empty file or write error).")

    except UnidentifiedImageError:
        app.logger.error(f"Job {job_id} failed: UnidentifiedImageError - Corrupt file?")
        with JOB_LOCK:
            job['status'] = 'failed'
            job['log'] = "The file is not a valid image or is corrupt."
    except Exception as e:
        app.logger.error(f"Job {job_id} failed: {repr(e)}")
        with JOB_LOCK:
            job['status'] = 'failed'
            job['log'] = f"Error processing image: {str(e)[:100]}."
    finally:
        # Crucial: delete the original uploaded file
        cleanup_temp_file(input_filepath)


# 3. Cleanup Task


def cleanup_old_files():
    cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=CLEANUP_INTERVAL_MINUTES)
    for directory in [TEMP_DIR, OUTPUT_FOLDER]:
        if not os.path.exists(directory): continue
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            try:
                if os.path.isfile(filepath):
                    mtime = datetime.fromtimestamp(os.path.getmtime(filepath), timezone.utc)
                    if mtime < cutoff_time: os.remove(filepath)
            except Exception as e:
                app.logger.warning(f"Cleanup error on {filepath}: {e}")

    with JOB_LOCK:
        to_delete = []
        for jid, job in JOB_QUEUE.items():
            if job.get('end_time'):
                end_dt = datetime.fromisoformat(job['end_time'])
                if end_dt < cutoff_time: to_delete.append(jid)
            elif datetime.fromisoformat(job['start_time']) < (cutoff_time - timedelta(minutes=60)):
                to_delete.append(jid)
        for jid in to_delete: del JOB_QUEUE[jid]


def start_cleanup_scheduler():
    def cleanup_loop():
        while True:
            time.sleep(CLEANUP_LOOP_SECONDS)
            cleanup_old_files()

    t = threading.Thread(target=cleanup_loop, daemon=True)
    t.start()


# 4. API Endpoints


@app.route('/api/config', methods=['GET'])
def get_config():
    limit = int(app.config["MAX_CONTENT_LENGTH"] / (1024 * 1024))
    conversions_list = [{'key': k, **v} for k, v in CONVERSION_MAP.items()]
    return jsonify({
        "max_upload_mb": limit,
        "conversions": conversions_list
    })


@app.route('/api/jobs', methods=['POST'])
@limiter.limit("30 per minute")
def create_job():
    if 'file' not in request.files:
        return jsonify({"msg": "Missing file"}), 400

    file = request.files['file']
    conversion_key = request.form.get('conversion_key')

    try:
        settings = json.loads(request.form.get('settings', '{}'))
    except json.JSONDecodeError:
        return jsonify({"msg": "Invalid settings JSON"}), 400

    if not file.filename or not conversion_key:
        return jsonify({"msg": "Invalid data"}), 400
    if conversion_key not in CONVERSION_MAP:
        return jsonify({"msg": "Unknown format"}), 400

    # 1. SECURITY: Read file stream into memory for initial inspection
    file_stream = io.BytesIO(file.read())
    file.stream.seek(0)

    # 2. SECURITY: MIME check (using python-magic if available)
    mime_type = get_file_magic_mime(file_stream)
    is_valid_image_mime = mime_type.startswith('image/')

    # Block common non-image file types
    if not is_valid_image_mime and mime_type in ['application/pdf', 'application/zip', 'text/plain']:
        return jsonify({"msg": f"File type {mime_type} is explicitly blocked."}), 403

    # Block unknown MIME types that aren't the generic octet stream (relying on PIL later)
    if not is_valid_image_mime and mime_type != 'application/octet-stream':
        return jsonify({"msg": f"Invalid file type: {mime_type}"}), 403

    # 3. Save file temporarily
    job_uuid = secrets.token_urlsafe(12)
    input_original_filename = custom_secure_filename(file.filename)
    input_filepath = os.path.join(TEMP_DIR, f"{job_uuid}_{input_original_filename}")

    try:
        file.save(input_filepath)
    except Exception as e:
        app.logger.error(f"File save failed: {e}")
        return jsonify({"msg": "Internal save error"}), 500

    # 4. SECURITY: PIL Structural Verification (after saving)
    try:
        img = Image.open(input_filepath)
        img.verify()
        img.close()
    except UnidentifiedImageError:
        app.logger.warning(f"Security Warning: File verification failed (Not recognized).")
        cleanup_temp_file(input_filepath)
        return jsonify({"msg": "File is not a valid image format."}), 415
    except Exception as e:
        app.logger.error(f"PIL verification failed: {e}")
        cleanup_temp_file(input_filepath)
        return jsonify({"msg": f"Image structural error: {str(e)[:100]}"}), 415

    # 5. Create Job and Queue
    with JOB_LOCK:
        JOB_QUEUE[job_uuid] = {
            'job_id': job_uuid,
            'input_original_filename': file.filename,
            'input_filepath': input_filepath,
            'output_filepath': None,
            'conversion_key': conversion_key,
            'settings': settings,
            'status': 'queued',
            'progress': 0,
            'start_time': datetime.now(timezone.utc).isoformat(),
        }

    WORKER_POOL.submit(run_image_job, job_uuid)
    return jsonify(job_id=job_uuid, status='queued'), 202


@app.route('/api/status/<string:job_uuid>', methods=['GET'])
def get_job_status(job_uuid):
    with JOB_LOCK:
        job = JOB_QUEUE.get(job_uuid)
    if not job: return jsonify({"msg": "Not found"}), 404
    job_data = job.copy()
    if job_data['status'] == 'completed' and job_data['output_filepath']:
        job_data['download_url'] = url_for('download_file', job_uuid=job_uuid, _external=True)
        # Do NOT expose server file paths to the client
        job_data.pop('input_filepath', None)
        job_data.pop('output_filepath', None)
    return jsonify(job_data), 200


@app.route('/api/download/<string:job_uuid>', methods=['GET'])
def download_file(job_uuid):
    with JOB_LOCK:
        job = JOB_QUEUE.get(job_uuid)
    if not job or job['status'] != 'completed' or not job['output_filepath']:
        return jsonify({"msg": "Unavailable"}), 404
    output_filepath = job['output_filepath']
    details = CONVERSION_MAP.get(job['conversion_key'])
    try:
        return send_file(
            output_filepath,
            mimetype=details['mime'],
            as_attachment=True,
            download_name=os.path.basename(output_filepath)
        )
    except FileNotFoundError:
        return jsonify({"msg": "File expired or deleted"}), 410


if __name__ == '__main__':
    start_cleanup_scheduler()
    #app.run(debug=os.getenv('FLASK_DEBUG', 'False') == 'True', host='0.0.0.0', port=5000)
