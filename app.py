import webbrowser
import os
from flask import Flask, Response, render_template, url_for, send_from_directory, request, redirect, jsonify
import torch
import platform
import pathlib
from PIL import Image
import datetime
import cv2
import numpy as np
import base64
import threading
import io
import time
import warnings
import json

warnings.filterwarnings('ignore')

if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

app = Flask(__name__)

TEMP_DIR = "temp"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

MAX_VIDEO_FILES = 4

DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"
MODEL_PATH = str(pathlib.Path('model/pytorch/best.pt').resolve())
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)
model.eval()

processing_tasks = {}

def initialize_task(task_id):
    if task_id not in processing_tasks:
        processing_tasks[task_id] = {
            'status': 'pending',
            'url': None,
            'stream_active': False,
            'lock': threading.Lock(),
            'timestamp': datetime.datetime.now()
        }

def cleanup_old_tasks():
    current_time = datetime.datetime.now()
    for task_id in list(processing_tasks.keys()):
        with processing_tasks[task_id]['lock']:
            task = processing_tasks[task_id]
            if task['status'] in ['completed', 'error']:
                if (current_time - task['timestamp']).total_seconds() > 3600:
                    del processing_tasks[task_id]

def clear_temp_directory():
    video_extensions = ('.mp4', '.avi', '.mov')
    all_files = os.listdir(TEMP_DIR)
    
    video_files = [f for f in all_files if f.lower().endswith(video_extensions)]
    num_video_files = len(video_files)
    
    if num_video_files >= MAX_VIDEO_FILES:
        print(f"Number of video files ({num_video_files}) exceeds limit ({MAX_VIDEO_FILES}). Clearing all video files.")
        for file in video_files:
            file_path = os.path.join(TEMP_DIR, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
    else:
        print(f"Number of video files ({num_video_files}) is within limit ({MAX_VIDEO_FILES}). No action taken.")

def process_and_stream_video(input_path, output_path, task_id, output_url):
    initialize_task(task_id)

    with processing_tasks[task_id]['lock']:
        processing_tasks[task_id].update({'status': 'streaming', 'stream_active': True})

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        with processing_tasks[task_id]['lock']:
            processing_tasks[task_id].update({'status': 'error', 'url': None, 'stream_active': False})
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while True:
        with processing_tasks[task_id]['lock']:
            if not processing_tasks[task_id]['stream_active']:
                break

        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        results = model(img)
        results.render()
        frame_processed = cv2.cvtColor(np.array(results.ims[0]), cv2.COLOR_RGB2BGR)

        out.write(frame_processed)
        ret, buffer = cv2.imencode('.jpg', frame_processed)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    out.release()
    if os.path.exists(output_path):
        with processing_tasks[task_id]['lock']:
            processing_tasks[task_id].update({
                'status': 'completed',
                'url': output_url,
                'stream_active': False
            })
    else:
        with processing_tasks[task_id]['lock']:
            processing_tasks[task_id].update({'status': 'error', 'url': None, 'stream_active': False})

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        clear_temp_directory()
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file or file.filename == '':
            return redirect(request.url)

        
        timestamp = datetime.datetime.now().strftime(DATETIME_FORMAT)
        task_id = f"task_{timestamp}"
        initialize_task(task_id)

        filename = file.filename.lower()

        if filename.endswith(('.png', '.jpg', '.jpeg')):
            try:
                img_bytes = file.read()
                img = Image.open(io.BytesIO(img_bytes))
                results = model([img])
                results.render()
                img_processed = Image.fromarray(results.ims[0])
                buffered = io.BytesIO()
                img_processed.save(buffered, format="PNG")
                img_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
                return render_template("index.html", img_data=img_data)
            except Exception as e:
                return f"Error processing image: {str(e)}", 500

        elif filename.endswith(('.mp4', '.avi', '.mov')):
            input_path = os.path.join(TEMP_DIR, f"input_{timestamp}.mp4")
            file.save(input_path)
            if not os.path.exists(input_path):
                return "Error: File could not be saved", 500
            return redirect(url_for('index', task_id=task_id))

    task_id = request.args.get('task_id')
    cleanup_old_tasks()
    return render_template("index.html", task_id=task_id)

@app.route('/stream/<task_id>')
def stream(task_id):

    initialize_task(task_id)
    input_path = os.path.join(TEMP_DIR, f"input_{task_id.replace('task_', '')}.mp4")
    output_path = os.path.join(TEMP_DIR, f"output_{task_id.replace('task_', '')}.mp4")
    
    if not os.path.exists(input_path):
        return f"Input file not found: {input_path}", 404

    output_url = url_for('serve_video', filename=os.path.basename(output_path), _external=False)

    if processing_tasks[task_id]['status'] == 'pending':
        thread = threading.Thread(target=lambda: list(process_and_stream_video(input_path, output_path, task_id, output_url)))
        thread.start()

    return Response(process_and_stream_video(input_path, output_path, task_id, output_url),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status-stream/<task_id>')
def status_stream(task_id):
    def event_stream():
        initialize_task(task_id)
        while True:
            with processing_tasks[task_id]['lock']:
                task = processing_tasks[task_id]
                status_data = {'status': task['status'], 'url': task['url']}
                yield f"data: {json.dumps(status_data)}\n\n"
                if task['status'] in ['completed', 'error']:
                    break
            time.sleep(1)
    return Response(event_stream(), mimetype="text/event-stream")

@app.route('/temp/<path:filename>')
def serve_video(filename):
    file_path = os.path.join(TEMP_DIR, filename)
    if os.path.exists(file_path):
        return send_from_directory(TEMP_DIR, filename, mimetype='video/mp4')
    else:
        return "File not found", 404

if __name__ == '__main__':
    host = '127.0.0.1'
    port = 5000
    url = f'http://{host}:{port}'
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        webbrowser.open(url)
    app.run(debug=True, host=host, port=port)
