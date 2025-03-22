import os
import platform
import pathlib
import datetime
import cv2
import numpy as np
import base64
import threading
import time
import io
import warnings
from uuid import uuid4
from flask import Flask, render_template, request, jsonify, send_from_directory, session
from flask_session import Session
from flask_socketio import SocketIO, emit
from PIL import Image
import webbrowser
from ultralytics import YOLO

warnings.filterwarnings('ignore')

if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret-key'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

socketio = SocketIO(app, async_mode='threading')

TEMP_DIR = "temp"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"
MODEL_PATH = str(pathlib.Path('model/pytorch/best_v11.pt').resolve())
model = YOLO(MODEL_PATH)
model.fuse()

class_colors = {}
for cls_id in model.names.keys():
    class_colors[cls_id] = tuple(np.random.randint(0, 255, 3).tolist())

processing_tasks = {}

def create_heatmap(frame, detections, heatmap_accumulator):
    h, w = frame.shape[:2]
    for det in detections:
        x_min, y_min, x_max, y_max = map(int, det[:4])
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        
        if 0 <= center_y < h and 0 <= center_x < w:
            heatmap_accumulator[center_y, center_x] += 10

    heatmap_smoothed = cv2.GaussianBlur(heatmap_accumulator, (15, 15), 5)
    heatmap_normalized = cv2.normalize(heatmap_smoothed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_boosted = np.power(heatmap_normalized / 255.0, 0.5) * 255
    heatmap_boosted = heatmap_boosted.astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_boosted, cv2.COLORMAP_HOT)

    overlay = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)
    return overlay

def draw_detections(frame, detections, display_options, class_colors):
    for det in detections:
        x_min, y_min, x_max, y_max, conf, cls = det
        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
        cls_id = int(cls)
        cls_name = model.names[cls_id]
        color = class_colors[cls_id]

        if display_options['bbox']:
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 3)

        if display_options['class_name'] or display_options['confidence']:
            label = cls_name if display_options['class_name'] else ""
            if display_options['confidence']:
                label += f" {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(y_min - 10, label_size[1] + 10)
            cv2.rectangle(frame, 
                          (x_min, label_ymin - label_size[1] - 10), 
                          (x_min + label_size[0] + 10, label_ymin), 
                          color, cv2.FILLED)
            cv2.putText(frame, label, 
                        (x_min + 5, label_ymin - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame

def initialize_task(task_id, session_id, sid):
    if task_id not in processing_tasks:
        processing_tasks[task_id] = {
            'status': 'pending',
            'url': None,
            'stream_active': False,
            'paused': False,
            'lock': threading.Lock(),
            'timestamp': datetime.datetime.now(),
            'session_id': session_id,
            'cap': None,
            'out': None,
            'sid': sid,
            'roi': None,
            'heatmap_accumulator': None,
            'show_heatmap': False,
            'display_options': {
                'class_name': True,
                'confidence': False,
                'bbox': True
            }
        }

def cleanup_user_tasks(session_id):
    for task_id in list(processing_tasks.keys()):
        with processing_tasks[task_id]['lock']:
            task = processing_tasks[task_id]
            if task['session_id'] == session_id:
                task['stream_active'] = False
                if task['cap'] and task['cap'].isOpened():
                    task['cap'].release()
                    task['out'].release()
                time.sleep(1)
                delete_task_videos(task_id)
                del processing_tasks[task_id]

def delete_task_videos(task_id):
    input_path = os.path.join(TEMP_DIR, f"input_{task_id.replace('task_', '')}.mp4")
    output_path = os.path.join(TEMP_DIR, f"output_{task_id.replace('task_', '')}.mp4")
    for path in [input_path, output_path]:
        if os.path.exists(path):
            try:
                os.unlink(path)
            except PermissionError as e:
                print(e)

def process_and_stream_video(input_path, output_path, task_id, output_url, sid):
    with processing_tasks[task_id]['lock']:
        processing_tasks[task_id].update({'status': 'streaming', 'stream_active': True})
        socketio.emit('status_update', {'status': 'streaming', 'url': None}, room=sid)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        with processing_tasks[task_id]['lock']:
            processing_tasks[task_id].update({'status': 'error', 'url': None, 'stream_active': False})
        socketio.emit('status_update', {'status': 'error', 'url': None}, room=sid)
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    heatmap_accumulator = np.zeros((frame_height, frame_width), dtype=np.float32)

    last_processed_frame = 0

    with processing_tasks[task_id]['lock']:
        processing_tasks[task_id]['cap'] = cap
        processing_tasks[task_id]['out'] = out
        processing_tasks[task_id]['heatmap_accumulator'] = heatmap_accumulator

    first_frame = True

    while True:
        if task_id not in processing_tasks:
            break
        with processing_tasks[task_id]['lock']:
            if task_id not in processing_tasks:
                break
            if not processing_tasks[task_id]['stream_active']:
                break
            if processing_tasks[task_id]['paused']:
                socketio.emit('status_update', {'status': 'paused', 'url': None}, room=sid)
                time.sleep(0.1)
                continue
            roi = processing_tasks[task_id]['roi']
            show_heatmap = processing_tasks[task_id]['show_heatmap']
            display_options = processing_tasks[task_id]['display_options']
            heatmap_accumulator = processing_tasks[task_id]['heatmap_accumulator']

        if last_processed_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, last_processed_frame)

        success, frame = cap.read()
        if not success:
            break

        frame_to_send = frame.copy()

        if roi:
            x = int(roi['x'])
            y = int(roi['y'])
            w = int(roi['width'])
            h = int(roi['height'])
            x = max(0, min(x, frame_width - 1))
            y = max(0, min(y, frame_height - 1))
            w = min(w, frame_width - x)
            h = min(h, frame_height - y)
            roi_frame = frame[y:y+h, x:x+w]
            if roi_frame.size > 0:
                results = model(roi_frame, verbose=False)
                detections = results[0].boxes.data.cpu().numpy()
                frame_to_send_roi = roi_frame.copy()
                frame_to_send_roi = draw_detections(frame_to_send_roi, detections, display_options, class_colors)
                frame_to_send = cv2.resize(frame_to_send_roi, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)
                if show_heatmap:
                    for det in detections:
                        det[:4] = det[:4] * [w / frame_width, h / frame_height, w / frame_width, h / frame_height]
                        det[:2] += [x, y]
                        det[2:4] += [x, y]
                    frame_to_send = create_heatmap(frame_to_send, detections, heatmap_accumulator)
                out.write(frame_to_send)
        else:
            results = model(frame, verbose=False)
            detections = results[0].boxes.data.cpu().numpy()
            frame_to_send = draw_detections(frame_to_send, detections, display_options, class_colors)
            if show_heatmap:
                frame_to_send = create_heatmap(frame_to_send, detections, heatmap_accumulator)
            out.write(frame_to_send)
        
        ret, buffer = cv2.imencode('.jpg', frame_to_send)
        frame_bytes = base64.b64encode(buffer.tobytes()).decode('utf-8')
        if first_frame:
            socketio.emit('video_frame', {
                'task_id': task_id,
                'frame': frame_bytes,
                'frame_width': frame_width,
                'frame_height': frame_height
            }, room=sid)
            first_frame = False
        else:
            socketio.emit('video_frame', {'task_id': task_id, 'frame': frame_bytes}, room=sid)

        last_processed_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    cap.release()
    out.release()
    if task_id not in processing_tasks:
        return
    if os.path.exists(output_path):
        with processing_tasks[task_id]['lock']:
            processing_tasks[task_id].update({
                'status': 'completed',
                'url': output_url,
                'stream_active': False,
                'cap': None,
                'out': None
            })
        socketio.emit('status_update', {'status': 'completed', 'url': output_url}, room=sid)
    else:
        with processing_tasks[task_id]['lock']:
            processing_tasks[task_id].update({'status': 'error', 'url': None, 'stream_active': False})
        socketio.emit('status_update', {'status': 'error', 'url': None}, room=sid)

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'session_id' not in session:
        session['session_id'] = str(uuid4())
    session_id = session['session_id']
    cleanup_user_tasks(session_id)

    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        if not file or file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        timestamp = datetime.datetime.now().strftime(DATETIME_FORMAT)
        task_id = f"task_{timestamp}"
        initialize_task(task_id, session_id, request.sid if hasattr(request, 'sid') else None)

        filename = file.filename.lower()
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            try:
                img_bytes = file.read()
                img = Image.open(io.BytesIO(img_bytes))
                img_array = np.array(img)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                results = model(img_array)
                img_processed = results[0].plot()
                img_processed_rgb = cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_processed_rgb)
                buffered = io.BytesIO()
                img_pil.save(buffered, format="PNG")
                img_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
                return jsonify({"img_data": img_data})
            except Exception as e:
                return jsonify({"error": f"Error processing image: {str(e)}"}), 500
        elif filename.endswith(('.mp4', '.avi', '.mov')):
            input_path = os.path.join(TEMP_DIR, f"input_{timestamp}.mp4")
            output_path = os.path.join(TEMP_DIR, f"output_{timestamp}.mp4")
            file.save(input_path)
            if not os.path.exists(input_path):
                return jsonify({"error": "File could not be saved"}), 500
            return jsonify({"task_id": task_id})

    return render_template("index.html")

@socketio.on('start_stream')
def handle_start_stream(data):
    task_id = data['task_id']
    if task_id not in processing_tasks:
        emit('status_update', {'status': 'error', 'url': None, 'error': 'Task not found'})
        return
    with processing_tasks[task_id]['lock']:
        processing_tasks[task_id]['sid'] = request.sid
    input_path = os.path.join(TEMP_DIR, f"input_{task_id.replace('task_', '')}.mp4")
    output_path = os.path.join(TEMP_DIR, f"output_{task_id.replace('task_', '')}.mp4")
    output_url = f"/temp/{os.path.basename(output_path)}"
    socketio.start_background_task(process_and_stream_video, input_path, output_path, task_id, output_url, request.sid)

@socketio.on('toggle_pause')
def handle_toggle_pause(data):
    task_id = data['task_id']
    if task_id not in processing_tasks:
        emit('status_update', {'status': 'error', 'url': None, 'error': 'Task not found'})
        return
    with processing_tasks[task_id]['lock']:
        if processing_tasks[task_id]['status'] != 'streaming':
            emit('status_update', {'status': 'error', 'url': None, 'error': 'Task is not streaming'})
            return
        processing_tasks[task_id]['paused'] = not processing_tasks[task_id]['paused']
        new_state = 'paused' if processing_tasks[task_id]['paused'] else 'streaming'
        emit('status_update', {'status': new_state, 'url': None})

@socketio.on('update_roi')
def handle_update_roi(data):
    task_id = data['task_id']
    roi = data['roi']
    if task_id in processing_tasks:
        with processing_tasks[task_id]['lock']:
            if task_id not in processing_tasks:
                return
            if processing_tasks[task_id]['status'] == 'streaming':
                processing_tasks[task_id]['roi'] = roi

@socketio.on('toggle_heatmap')
def handle_toggle_heatmap(data):
    task_id = data['task_id']
    if task_id not in processing_tasks:
        emit('status_update', {'status': 'error', 'url': None, 'error': 'Task not found'})
        return
    with processing_tasks[task_id]['lock']:
        if processing_tasks[task_id]['status'] != 'streaming':
            emit('status_update', {'status': 'error', 'url': None, 'error': 'Task is not streaming'})
            return
        processing_tasks[task_id]['show_heatmap'] = not processing_tasks[task_id]['show_heatmap']
        state = 'on' if processing_tasks[task_id]['show_heatmap'] else 'off'
        emit('heatmap_status', {'task_id': task_id, 'state': state})

@socketio.on('update_display_options')
def handle_update_display_options(data):
    task_id = data['task_id']
    if task_id not in processing_tasks:
        emit('status_update', {'status': 'error', 'url': None, 'error': 'Task not found'})
        return
    with processing_tasks[task_id]['lock']:
        if processing_tasks[task_id]['status'] not in ['streaming', 'paused']:
            emit('status_update', {'status': 'error', 'url': None, 'error': 'Task is not active'})
            return
        processing_tasks[task_id]['display_options'] = data['options']
        emit('display_options_updated', {'task_id': task_id, 'options': data['options']})

@app.route('/temp/<path:filename>')
def serve_video(filename):
    file_path = os.path.join(TEMP_DIR, filename)
    if os.path.exists(file_path):
        return send_from_directory(TEMP_DIR, filename, mimetype='video/mp4')
    return jsonify({"error": "File not found"}), 404

if __name__ == '__main__':
    host = '127.0.0.1'
    port = 5000
    url = f'http://{host}:{port}'
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        webbrowser.open(url)
    socketio.run(app, debug=True, host=host, port=port)