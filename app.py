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

from flask import Flask, request, jsonify, send_file, url_for, render_template
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageDraw, ImageFont # Added ImageDraw, ImageFont for new features
import math # Added for Color Replace distance calculation

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

# Helper for Color Replacement
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def color_distance(c1, c2):
    """Euclidean distance between two RGB colors."""
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(c1, c2)]))

# 2. Image Processing Workers

def run_processing_job(job_id):
    """Routes the job to the correct worker function (only convert remains)."""
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
    """Worker thread for image conversion and all editing features."""
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
        
        # 1. Open Image
        img = Image.open(input_filepath)

        # Handle animations (seek 0 unless GIF/WEBP target)
        if getattr(img, "is_animated", False) and target_format not in ['GIF', 'WEBP']:
            img.seek(0)
            
        # Ensure we have a working image mode for all features
        if img.mode == 'P':
            img = img.convert('RGBA' if 'transparency' in img.info else 'RGB')
            
        # Ensure RGBA for transparency-based features
        if img.mode != 'RGBA': 
            img = img.convert('RGBA')

        # A. RESIZING (Existing Logic)
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
            job['progress'] = 30

        # B. COLOR REPLACEMENT (NEW FEATURE)
        target_hex = settings.get('color_replace_target')
        new_hex = settings.get('color_replace_new')
        tolerance = int(settings.get('color_replace_tolerance', 0))

        if target_hex and new_hex and tolerance > 0:
            try:
                target_rgb = hex_to_rgb(target_hex)
                new_rgb = hex_to_rgb(new_hex)
                
                # Check for alpha channel if image is RGBA
                target_rgb = target_rgb + (255,) if img.mode == 'RGBA' and len(target_rgb) == 3 else target_rgb
                new_rgb = new_rgb + (255,) if img.mode == 'RGBA' and len(new_rgb) == 3 else new_rgb
                
                data = list(img.getdata())
                new_data = []

                # Max distance for RGB colors is sqrt(255^2 * 3) ~ 441.7
                max_tolerance = 442 # Use 442 to represent 100% tolerance visually

                # Normalize tolerance from 0-100 to 0-442
                normalized_tolerance = (tolerance / 100.0) * max_tolerance

                for pixel in data:
                    if color_distance(pixel[:3], target_rgb[:3]) <= normalized_tolerance:
                        new_data.append(new_rgb)
                    else:
                        new_data.append(pixel)
                
                img.putdata(new_data)
            except Exception as e:
                app.logger.error(f"Color replacement failed: {e}")
                
        with JOB_LOCK:
            job['progress'] = 40
            
        # C. NEW IMAGE EDITING FEATURES (Transforms, Enhancements, Filters)

        # C1. FLIP & ROTATE (Existing Logic)
        if settings.get('flip_h'): 
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if settings.get('flip_v'): 
            img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        
        rotate_angle = settings.get('rotate_angle', 0) 
        try:
            angle = int(rotate_angle) % 360
            if angle != 0:
                img = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
        except ValueError:
            pass 

        # Ensure image is in RGB/RGBA for enhancements and filters
        if img.mode not in ('RGB', 'RGBA'): img = img.convert('RGB')

        # C2. ENHANCEMENTS (Brightness, Contrast, Sharpen, Blur - Existing Logic)
        
        brightness_factor = float(settings.get('brightness', 100)) / 100.0
        if brightness_factor != 1.0:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness_factor)

        contrast_factor = float(settings.get('contrast', 100)) / 100.0
        if contrast_factor != 1.0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast_factor)
            
        sharpness_factor = float(settings.get('sharpen', 0)) / 100.0
        if sharpness_factor > 0:
            enhancer = ImageEnhance.Sharpness(img)
            enhance_val = 1.0 + (sharpness_factor * 2.0 / 100.0)
            img = enhancer.enhance(enhance_val)

        blur_level = int(settings.get('blur', 0))
        if blur_level > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_level / 10.0))

        # C3. FILTERS (Grayscale, Sepia, Invert - Existing Logic)
        if settings.get('filter_grayscale'): 
            img = img.convert('L').convert('RGB')
        
        if settings.get('filter_sepia'): 
            if img.mode != 'RGB': img = img.convert('RGB')
            r, g, b = img.split()
            # Simple sepia tone approximation
            r_sepia = r.point(lambda p: min(255, int(p * 0.393 + g.getpixel((0,0)) * 0.769 + b.getpixel((0,0)) * 0.189)))
            g_sepia = g.point(lambda p: min(255, int(r.getpixel((0,0)) * 0.349 + p * 0.686 + b.getpixel((0,0)) * 0.168)))
            b_sepia = b.point(lambda p: min(255, int(r.getpixel((0,0)) * 0.272 + g.getpixel((0,0)) * 0.534 + p * 0.131)))
            img = Image.merge('RGB', (r_sepia, g_sepia, b_sepia))

        if settings.get('filter_invert'): 
            if img.mode not in ('RGB', 'RGBA'): img = img.convert('RGB')
            img = ImageOps.invert(img)
            
        # Ensure RGBA for final steps
        if img.mode != 'RGBA': img = img.convert('RGBA')

        with JOB_LOCK:
            job['progress'] = 60
            
        # D. WATERMARK (NEW FEATURE)
        watermark_text = settings.get('watermark_text', '').strip()
        watermark_size = int(settings.get('watermark_size', 20))
        watermark_color = settings.get('watermark_color', '#FFFFFF')
        
        if watermark_text:
            try:
                draw = ImageDraw.Draw(img)
                font_path = os.path.join(BASE_DIR, "Arial.ttf") # Attempt to use a common font or fall back
                try:
                    # Look for a font file (Pillow needs a font file path for draw.text)
                    font = ImageFont.truetype("arial.ttf", watermark_size)
                except IOError:
                    # Fallback to default font if arial.ttf is not found in search path
                    font = ImageFont.load_default()
                    
                # Calculate text size (using default font as a safer fallback)
                bbox = draw.textbbox((0, 0), watermark_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # Position (bottom right corner, 20px padding)
                x = img.width - text_width - 20
                y = img.height - text_height - 20
                
                draw.text((x, y), watermark_text, fill=watermark_color, font=font)
            except Exception as e:
                app.logger.error(f"Watermark failed: {e}")

        with JOB_LOCK:
            job['progress'] = 70
            
        # E. CORNER ROUNDING & BORDER (Fix for smooth border)
        border_size = int(settings.get('border_size', 0))
        border_color = settings.get('border_color', '#000000')
        corner_radius = int(settings.get('corner_radius', 0)) # Now a global setting

        if corner_radius > 0:
            width, height = img.size
            
            # Limit radius to half the shortest side
            safe_radius = min(corner_radius, min(width, height) // 2)
            
            # 1. Create a mask with rounded corners
            mask = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(mask)
            draw.rounded_rectangle((0, 0, width, height), safe_radius, fill=255)
            
            # 2. Apply the mask to the image (making the image itself rounded)
            img.putalpha(mask) 

        # 3. Apply Border (if border_size > 0 and corner_radius > 0, the border will be smooth)
        if border_size > 0:
            border_color_rgb = hex_to_rgb(border_color)
            
            # Create a new image for the border background
            bordered_img_size = (img.width + 2 * border_size, img.height + 2 * border_size)
            border_bg = Image.new('RGBA', bordered_img_size, border_color_rgb + (255,))
            
            # Paste the (now potentially rounded) image onto the border background
            border_bg.paste(img, (border_size, border_size), img)
            img = border_bg
            
            # NOTE: If the image is rounded, the border is now implicitly rounded 
            # by the original image's alpha channel cut-out.

        # F. OPACITY ADJUSTMENT (Existing Logic)
        try:
            opacity_val = int(settings.get('opacity', 100))
        except (ValueError, TypeError):
            opacity_val = 100

        if 0 <= opacity_val < 100:
            # Opacity is already handled since we forced RGBA mode
            alpha = img.split()[3]
            factor = opacity_val / 100.0
            alpha = alpha.point(lambda p: int(p * factor))
            img.putalpha(alpha)

        with JOB_LOCK:
            job['progress'] = 80

        # G. COLOR MODE & FORMAT HANDLING (Original logic)
        if target_format in ['JPEG', 'BMP']:
            if img.mode in ('RGBA', 'LA'):
                # Handle conversion to RGB by pasting onto white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')

        elif target_format == 'ICO':
            img = ImageOps.contain(img, (256, 256))

        # H. SAVING & QUALITY (Original logic)
        save_kwargs = {}
        if details['supports_quality']:
            try:
                q = int(settings.get('quality', 90))
                save_kwargs['quality'] = max(1, min(100, q))
            except (ValueError, TypeError):
                save_kwargs['quality'] = 90

        if target_format == 'GIF':
            save_kwargs['optimize'] = True
            if getattr(img, "is_animated", False):
                save_kwargs['save_all'] = True

        img.save(output_filepath, format=target_format, **save_kwargs)
        img.close() 

        if os.path.exists(output_filepath) and os.path.getsize(output_filepath) > 0:
            with JOB_LOCK:
                job['status'] = 'completed'
                job['progress'] = 100
                job['end_time'] = datetime.now(timezone.utc).isoformat()
        else:
            raise Exception("File save failed (empty file).")

    except UnidentifiedImageError:
         app.logger.error(f"Job {job_id} failed: The file could not be opened as an image.")
         with JOB_LOCK:
            job['status'] = 'failed'
            job['log'] = "The file is not a valid or supported image file."
    except Exception as e:
        app.logger.error(f"Job {job_id} failed: {repr(e)}")
        with JOB_LOCK:
            job['status'] = 'failed'
            job['log'] = f"Error processing image: {str(e)}"
    finally:
        if os.path.exists(input_filepath):
            try:
                os.remove(input_filepath)
            except OSError:
                pass


# 3. Cleanup Task
# (No changes here)


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
    mode = request.form.get('mode')
    conversion_key = request.form.get('conversion_key')

    try:
        settings = json.loads(request.form.get('settings', '{}'))
    except json.JSONDecodeError:
        # Improved error handling for invalid settings JSON
        app.logger.error("Received invalid settings JSON.")
        return jsonify({"msg": "Invalid settings JSON"}), 400

    if not file.filename or mode != 'convert':
        return jsonify({"msg": "Invalid job data or mode"}), 400

    if conversion_key not in CONVERSION_MAP:
        return jsonify({"msg": "Unknown conversion format"}), 400

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
        app.logger.error(f"File save failed: {e}")
        return jsonify({"msg": "Internal save error during file upload"}), 500

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

    WORKER_POOL.submit(run_processing_job, job_uuid)
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
    app.run(debug=True, port=5000, threaded=True)
