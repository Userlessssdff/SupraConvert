# app.py - Ultra Image Convert & Upscale Backend

import os
import secrets
import json
import time
import logging
from logging.handlers import RotatingFileHandler
import threading
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, request, jsonify, send_file, url_for, render_template
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv
from PIL import Image, ImageOps
import requests  # Required for HF API

# 0. Configuration & Initialization

load_dotenv()

# Logging Setup
handler = RotatingFileHandler('app.log', maxBytes=100000, backupCount=3)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    handlers=[handler, logging.StreamHandler()]
)

# --- Environment & Secrets ---
HF_TOKEN = os.getenv("HF_TOKEN")
# Using Swin2SR x2 - Good balance of speed and quality for free tier
HF_MODEL_URL = os.getenv("HF_MODEL_URL",
                         "https://api-inference.huggingface.co/models/caidas/swin2SR-classical-sr-x2-64")

if not HF_TOKEN:
    logging.warning("HF_TOKEN is missing in .env! Upscaling will fail.")

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
    file_stream.seek(0)
    header = file_stream.read(2048)
    file_stream.seek(0)
    try:
        from magic import from_buffer
        return from_buffer(header, mime=True)
    except ImportError:
        logging.warning("python-magic not installed, using fallback validation.")
        return 'application/octet-stream'
    except Exception as e:
        logging.error(f"Mime detection error: {e}")
        return 'application/octet-stream'


def upscale_image_remote(input_path, output_path):
    """Calls HF API to upscale image."""

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    with open(input_path, "rb") as f:
        data = f.read()

    # Retry logic for model loading (HF specific)
    retries = 5
    for attempt in range(retries):
        try:
            response = requests.post(HF_MODEL_URL, headers=headers, data=data)

            # If successful
            if response.status_code == 200:
                with open(output_path, "wb") as f_out:
                    f_out.write(response.content)
                return True, None

            # If model is loading, wait and retry
            error_data = response.json()
            if "error" in error_data and "loading" in error_data["error"].lower():
                wait_time = error_data.get("estimated_time", 5)
                logging.info(f"Model loading... waiting {wait_time}s")
                time.sleep(wait_time)
                continue
            else:
                return False, f"HF API Error: {error_data.get('error', 'Unknown error')}"

        except Exception as e:
            return False, str(e)

    return False, "Model took too long to load."


# 2. Image Processing Worker

def run_image_job(job_id):
    """Worker thread that handles both Conversion and Upscaling."""
    with JOB_LOCK:
        job = JOB_QUEUE.get(job_id)

    if not job: return

    with JOB_LOCK:
        job['status'] = 'running'
        job['progress'] = 10

    input_filepath = job['input_filepath']
    mode = job.get('mode', 'convert')  # 'convert' or 'upscale'
    settings = job.get('settings', {})

    # Common Setup
    base_name = os.path.splitext(os.path.basename(job['input_original_filename']))[0]

    try:
        # --- BRANCH: UPSCALING ---
        if mode == 'upscale':
            if not HF_TOKEN:
                raise Exception("Server configuration error: HF_TOKEN missing.")

            # 1. First, get the raw AI upscaled result
            raw_upscale_filename = f"temp_upscaled_{base_name}_{secrets.token_urlsafe(4)}.png"
            raw_upscale_path = os.path.join(TEMP_DIR, raw_upscale_filename)

            with JOB_LOCK:
                job['progress'] = 30  # Sending to API

            success, error_msg = upscale_image_remote(input_filepath, raw_upscale_path)
            if not success: raise Exception(error_msg)

            with JOB_LOCK:
                job['progress'] = 70  # Processing Result

            # 2. Now handle the resolution target (1080p, 2K, 4K)
            target_res = settings.get('upscale_res', 'original')  # '1080p', '2k', '4k', 'original'

            final_filename = f"upscaled_{target_res}_{base_name}_{secrets.token_urlsafe(4)}.png"
            final_filepath = os.path.join(OUTPUT_FOLDER, final_filename)

            # Map targets to max dimensions (width, height)
            res_map = {
                '1080p': (1920, 1080),
                '2k': (2560, 1440),
                '4k': (3840, 2160)
            }

            img = Image.open(raw_upscale_path)

            if target_res in res_map:
                target_w, target_h = res_map[target_res]
                # ImageOps.contain keeps aspect ratio but fits inside the box
                # method=Image.Resampling.LANCZOS for best quality
                img = ImageOps.contain(img, (target_w, target_h), method=Image.Resampling.LANCZOS)

            img.save(final_filepath, format="PNG")
            img.close()

            # Cleanup raw upscale
            try:
                os.remove(raw_upscale_path)
            except:
                pass

            with JOB_LOCK:
                job['output_filepath'] = final_filepath
                job['progress'] = 100
                job['status'] = 'completed'
                job['end_time'] = datetime.now(timezone.utc).isoformat()

        # --- BRANCH: CONVERSION (Original Logic) ---
        else:
            conversion_key = job['conversion_key']
            details = CONVERSION_MAP.get(conversion_key)
            if not details: raise Exception("Invalid conversion key")

            output_ext = details['ext']
            target_format = details['format']
            output_filename = f"processed_{base_name}_{secrets.token_urlsafe(4)}.{output_ext}"
            output_filepath = os.path.join(OUTPUT_FOLDER, output_filename)

            with JOB_LOCK:
                job['output_filepath'] = output_filepath
                job['progress'] = 20

            img = Image.open(input_filepath)

            # Animation Handling
            if getattr(img, "is_animated", False) and target_format not in ['GIF', 'WEBP']:
                img.seek(0)

            # Resizing
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
                if maintain_ar:
                    img = ImageOps.contain(img, (final_w, final_h), method=Image.Resampling.LANCZOS)
                else:
                    img = img.resize((final_w, final_h), resample=Image.Resampling.LANCZOS)

            with JOB_LOCK:
                job['progress'] = 50

            # Opacity
            try:
                opacity_val = int(settings.get('opacity', 100))
            except:
                opacity_val = 100
            if 0 <= opacity_val < 100:
                if img.mode != 'RGBA': img = img.convert('RGBA')
                alpha = img.split()[3]
                alpha = alpha.point(lambda p: int(p * (opacity_val / 100.0)))
                img.putalpha(alpha)

            # Color Mode
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

            # Saving
            save_kwargs = {}
            if details['supports_quality']:
                try:
                    q = int(settings.get('quality', 90))
                except:
                    q = 90
                save_kwargs['quality'] = max(1, min(100, q))

            if target_format == 'GIF':
                save_kwargs['optimize'] = True
                if getattr(img, "is_animated", False): save_kwargs['save_all'] = True

            img.save(output_filepath, format=target_format, **save_kwargs)
            img.close()

            with JOB_LOCK:
                job['status'] = 'completed'
                job['progress'] = 100
                job['end_time'] = datetime.now(timezone.utc).isoformat()

    except Exception as e:
        app.logger.error(f"Job {job_id} failed: {repr(e)}")
        with JOB_LOCK:
            job['status'] = 'failed'
            job['log'] = str(e)
    finally:
        if os.path.exists(input_filepath):
            try:
                os.remove(input_filepath)
            except OSError:
                pass


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
    return jsonify({
        "max_upload_mb": limit,
        "conversions": [{'key': k, **v} for k, v in CONVERSION_MAP.items()]
    })


@app.route('/api/jobs', methods=['POST'])
@limiter.limit("30 per minute")
def create_job():
    if 'file' not in request.files:
        return jsonify({"msg": "Missing file"}), 400

    file = request.files['file']
    mode = request.form.get('mode', 'convert')  # Get Mode

    # Validate Mode
    if mode not in ['convert', 'upscale']:
        return jsonify({"msg": "Invalid mode"}), 400

    # For convert mode, validate key
    conversion_key = request.form.get('conversion_key')
    if mode == 'convert' and (not conversion_key or conversion_key not in CONVERSION_MAP):
        return jsonify({"msg": "Unknown format"}), 400

    settings = {}
    try:
        settings = json.loads(request.form.get('settings', '{}'))
    except:
        pass

    if not file.filename: return jsonify({"msg": "Invalid file"}), 400

    mime_type = get_file_magic_mime(file.stream)
    if not mime_type.startswith('image/') and mime_type != 'application/octet-stream':
        return jsonify({"msg": f"Invalid file type: {mime_type}"}), 403

    job_uuid = secrets.token_urlsafe(12)
    input_filename = custom_secure_filename(file.filename)
    input_filepath = os.path.join(TEMP_DIR, input_filename)

    try:
        file.stream.seek(0)
        file.save(input_filepath)
    except Exception as e:
        app.logger.error(f"Save failed: {e}")
        return jsonify({"msg": "Internal save error"}), 500

    with JOB_LOCK:
        JOB_QUEUE[job_uuid] = {
            'job_id': job_uuid,
            'mode': mode,
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

    # Determine MIME
    mime = 'application/octet-stream'
    if job.get('mode') == 'upscale':
        mime = 'image/png'
    elif job.get('conversion_key'):
        mime = CONVERSION_MAP[job['conversion_key']]['mime']

    try:
        return send_file(
            output_filepath,
            mimetype=mime,
            as_attachment=True,
            download_name=os.path.basename(output_filepath)
        )
    except FileNotFoundError:
        return jsonify({"msg": "File expired or deleted"}), 410


if __name__ == '__main__':
    start_cleanup_scheduler()
    #app.run(debug=True, port=5000, threaded=True)