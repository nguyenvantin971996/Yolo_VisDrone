# Import thư viện để làm việc với hệ thống file
import os
# Import các thành phần Flask để xây dựng ứng dụng web
from flask import Flask, Response, render_template, request, jsonify, send_from_directory, session
# Import Flask-Session để quản lý phiên người dùng
from flask_session import Session
# Import để kiểm tra hệ điều hành
import platform
# Import để xử lý đường dẫn file đa nền tảng
import pathlib
# Import để xử lý ảnh
from PIL import Image
# Import để làm việc với thời gian
import datetime
# Import OpenCV để xử lý ảnh và video
import cv2
# Import NumPy để làm việc với mảng dữ liệu
import numpy as np
# Import để mã hóa dữ liệu thành base64
import base64
# Import để xử lý đa luồng
import threading
# Import để làm việc với luồng dữ liệu trong bộ nhớ
import io
# Import để tạo khoảng dừng thời gian
import time
# Import để tắt cảnh báo không cần thiết
import warnings
# Import để làm việc với dữ liệu JSON
import json
# Import để tự động mở trình duyệt
import webbrowser
# Import để tạo ID duy nhất
from uuid import uuid4
# Import mô hình YOLO từ Ultralytics
from ultralytics import YOLO

# Tắt tất cả cảnh báo để tránh log thừa
warnings.filterwarnings('ignore')

# Điều chỉnh pathlib để tương thích với Windows
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath  # Gán PosixPath thành WindowsPath trên Windows
else:
    pathlib.WindowsPath = pathlib.PosixPath  # Gán WindowsPath thành PosixPath trên Linux

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# Cấu hình khóa bí mật cho Flask
app.config['SECRET_KEY'] = 'secret-key'
# Cấu hình lưu session trên hệ thống file
app.config['SESSION_TYPE'] = 'filesystem'
# Kích hoạt Flask-Session cho ứng dụng
Session(app)

# Định nghĩa thư mục tạm để lưu file
TEMP_DIR = "temp"
# Kiểm tra và tạo thư mục tạm nếu chưa tồn tại
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)  # Tạo thư mục TEMP_DIR

# Định dạng chuỗi thời gian để tạo tên file/task_id
DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"
# Đường dẫn tới file mô hình YOLO (đã huấn luyện)
MODEL_PATH = str(pathlib.Path('model/pytorch/best_v11.pt').resolve())
# Tải mô hình YOLO từ đường dẫn
model = YOLO(MODEL_PATH)
# Tối ưu hóa mô hình YOLO cho hiệu suất
model.fuse()

# Dictionary để lưu trạng thái các task xử lý
processing_tasks = {}

# Hàm khởi tạo task mới với trạng thái ban đầu
def initialize_task(task_id, session_id):
    # Kiểm tra nếu task_id chưa tồn tại trong processing_tasks
    if task_id not in processing_tasks:
        # Tạo một task mới với các thuộc tính ban đầu
        processing_tasks[task_id] = {
            'status': 'pending',         # Trạng thái: đang chờ xử lý
            'url': None,                 # URL của file đầu ra (chưa có)
            'stream_active': False,      # Trạng thái stream: chưa hoạt động
            'paused': False,             # Trạng thái tạm dừng: chưa tạm dừng
            'lock': threading.Lock(),    # Khóa để đồng bộ luồng
            'timestamp': datetime.datetime.now(),  # Thời gian tạo task
            'session_id': session_id,    # ID phiên của người dùng
            'cap': None,                  # Đối tượng video capture (chưa có),
            'out': None
        }

# Hàm dọn dẹp các task của một session
def cleanup_user_tasks(session_id):
    # Duyệt qua tất cả task_id trong danh sách
    for task_id in list(processing_tasks.keys()):
        # Khóa để đảm bảo thread-safe
        with processing_tasks[task_id]['lock']:
            task = processing_tasks[task_id]  # Lấy thông tin task
            # Kiểm tra nếu task thuộc về session_id
            if task['session_id'] == session_id:
                task['stream_active'] = False  # Dừng stream
                # Kiểm tra và giải phóng tài nguyên video nếu đang mở
                if task['cap'] and task['cap'].isOpened():
                    task['cap'].release()     # Giải phóng video capture
                    task['cap'] = None         # Đặt lại cap về None
                    task['out'].release()
                    task['out'] = None 
                time.sleep(1)                  # Đợi 1 giây để hoàn tất
                delete_task_videos(task_id)    # Xóa file video liên quan
                del processing_tasks[task_id]  # Xóa task khỏi dictionary

def delete_task_videos(task_id):
    input_path = os.path.join(TEMP_DIR, f"input_{task_id.replace('task_', '')}.mp4")
    output_path = os.path.join(TEMP_DIR, f"output_{task_id.replace('task_', '')}.mp4")
    for path in [input_path, output_path]:
        if os.path.exists(path):
            try:
                os.unlink(path)
                print(f"Deleted: {path}")
            except PermissionError as e:
                print(e)

# Hàm xử lý và stream video
def process_and_stream_video(input_path, output_path, task_id, output_url):
    # Khởi tạo trạng thái ban đầu
    if task_id in processing_tasks:  # Kiểm tra trước khi truy cập
        with processing_tasks[task_id]['lock']:
            processing_tasks[task_id].update({'status': 'streaming', 'stream_active': True})
    else:
        return  # Thoát ngay nếu task_id không tồn tại

    # Mở file video đầu vào
    cap = cv2.VideoCapture(input_path)
    # Kiểm tra nếu không mở được video
    if not cap.isOpened():
        # Khóa để cập nhật trạng thái lỗi
        with processing_tasks[task_id]['lock']:
            # Cập nhật trạng thái thành error
            processing_tasks[task_id].update({'status': 'error', 'url': None, 'stream_active': False})
        return  # Thoát hàm

    # Lấy chiều rộng frame
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # Lấy chiều cao frame
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Lấy FPS của video
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Định nghĩa codec để ghi video
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    # Tạo đối tượng ghi video đầu ra
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Biến lưu vị trí frame cuối cùng đã xử lý
    last_processed_frame = 0
    # Biến lưu dữ liệu frame cuối cùng dưới dạng bytes
    last_frame_bytes = None

     # Lưu đối tượng capture vào task
    with processing_tasks[task_id]['lock']:
        processing_tasks[task_id]['cap'] = cap
        processing_tasks[task_id]['out'] = out

    # Vòng lặp xử lý từng frame
    while True:
        if task_id not in processing_tasks:
            break
        # Khóa để kiểm tra trạng thái task
        with processing_tasks[task_id]['lock']:
            if task_id not in processing_tasks:
                break
            # Nếu stream không còn hoạt động thì thoát
            if not processing_tasks[task_id]['stream_active']:
                break
            # Nếu task bị tạm dừng
            if processing_tasks[task_id]['paused']:
                # Nếu có frame cuối cùng, gửi lại frame đó
                if last_frame_bytes:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + last_frame_bytes + b'\r\n')
                time.sleep(0.1)  # Đợi 0.1 giây để giảm tải CPU
                continue

        # Đặt vị trí frame cần đọc nếu đã xử lý trước đó
        if last_processed_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, last_processed_frame)

        # Đọc frame từ video
        success, frame = cap.read()
        # Nếu không đọc được frame thì thoát
        if not success:
            break

        # Chạy mô hình YOLO để xử lý frame
        results = model(frame, verbose=False)
        # Vẽ kết quả lên frame
        frame_processed = results[0].plot()

        # Ghi frame đã xử lý vào file đầu ra
        out.write(frame_processed)
        # Mã hóa frame thành định dạng JPEG
        ret, buffer = cv2.imencode('.jpg', frame_processed)
        # Chuyển buffer thành bytes
        frame_bytes = buffer.tobytes()
        # Lưu frame cuối cùng
        last_frame_bytes = frame_bytes
        # Gửi frame qua stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Cập nhật vị trí frame cuối cùng đã xử lý
        last_processed_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    # Giải phóng tài nguyên video capture
    cap.release()
    # Giải phóng tài nguyên video writer
    out.release()
    # Kiểm tra nếu file đầu ra tồn tại
    if task_id not in processing_tasks:
        return
    if os.path.exists(output_path):
        # Khóa để cập nhật trạng thái hoàn tất
        with processing_tasks[task_id]['lock']:
            processing_tasks[task_id]['cap'] = None  # Đặt lại cap về None
            processing_tasks[task_id]['out'] = None
            # Cập nhật trạng thái thành completed và lưu URL
            processing_tasks[task_id].update({
                'status': 'completed',
                'url': output_url,
                'stream_active': False
            })
    else:
        # Khóa để cập nhật trạng thái lỗi
        with processing_tasks[task_id]['lock']:
            # Cập nhật trạng thái thành error
            processing_tasks[task_id].update({'status': 'error', 'url': None, 'stream_active': False})

# Route xử lý trang chính và upload file
@app.route('/', methods=['GET', 'POST'])
def index():
    # Tạo session_id nếu chưa tồn tại
    if 'session_id' not in session:
        session['session_id'] = str(uuid4())  # Tạo ID duy nhất bằng UUID

    # Lấy session_id từ session
    session_id = session['session_id']
    # Dọn dẹp các task cũ của session
    cleanup_user_tasks(session_id)
    # Nếu request là POST (upload file)
    if request.method == "POST":
        # Kiểm tra nếu không có file trong request
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400  # Trả về lỗi 400
        # Lấy file từ request
        file = request.files["file"]
        # Kiểm tra nếu file rỗng
        if not file or file.filename == '':
            return jsonify({"error": "No file selected"}), 400  # Trả về lỗi 400

        # Tạo timestamp để đặt tên file/task_id
        timestamp = datetime.datetime.now().strftime(DATETIME_FORMAT)
        # Tạo task_id từ timestamp
        task_id = f"task_{timestamp}"
        # Khởi tạo task mới
        initialize_task(task_id, session_id)

        # Lấy tên file và chuyển thành chữ thường
        filename = file.filename.lower()
        # Nếu file là ảnh
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            try:
                # Đọc dữ liệu ảnh dưới dạng bytes
                img_bytes = file.read()
                # Mở ảnh từ dữ liệu bytes
                img = Image.open(io.BytesIO(img_bytes))
                # Chuyển ảnh thành mảng NumPy
                img_array = np.array(img)
                # Chuyển đổi định dạng màu từ RGB sang BGR
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                # Chạy mô hình YOLO để xử lý ảnh
                results = model(img_array)
                # Vẽ kết quả lên ảnh
                img_processed = results[0].plot()
                # Chuyển đổi định dạng màu từ BGR sang RGB
                img_processed_rgb = cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB)
                # Chuyển mảng thành ảnh PIL
                img_pil = Image.fromarray(img_processed_rgb)
                # Tạo buffer để lưu ảnh
                buffered = io.BytesIO()
                # Lưu ảnh vào buffer dưới dạng PNG
                img_pil.save(buffered, format="PNG")
                # Mã hóa ảnh thành base64
                img_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
                # Trả về dữ liệu ảnh đã xử lý
                return jsonify({"img_data": img_data})
            except Exception as e:
                # Trả về lỗi nếu xử lý ảnh thất bại
                return jsonify({"error": f"Error processing image: {str(e)}"}), 500

        # Nếu file là video
        elif filename.endswith(('.mp4', '.avi', '.mov')):
            # Đường dẫn lưu file video đầu vào
            input_path = os.path.join(TEMP_DIR, f"input_{timestamp}.mp4")
            # Đường dẫn lưu file video đầu ra
            output_path = os.path.join(TEMP_DIR, f"output_{timestamp}.mp4")
            # Lưu file upload vào TEMP_DIR
            file.save(input_path)
            # Kiểm tra nếu file không lưu được
            if not os.path.exists(input_path):
                return jsonify({"error": "File could not be saved"}), 500  # Trả về lỗi 500

            # Trả về task_id để client theo dõi
            return jsonify({"task_id": task_id})

    # Render trang index.html với task_id
    return render_template("index.html")

# Route để stream video
@app.route('/stream/<task_id>')
def stream(task_id):
    # Kiểm tra nếu task_id không tồn tại
    if task_id not in processing_tasks:
        return jsonify({"error": "Task not found"}), 404  # Trả về lỗi 404

    # Đường dẫn file video đầu vào
    input_path = os.path.join(TEMP_DIR, f"input_{task_id.replace('task_', '')}.mp4")
    # Đường dẫn file video đầu ra
    output_path = os.path.join(TEMP_DIR, f"output_{task_id.replace('task_', '')}.mp4")
    
    # Kiểm tra nếu file đầu vào không tồn tại
    if not os.path.exists(input_path):
        return jsonify({"error": f"Input file not found: {input_path}"}), 404  # Trả về lỗi 404

    # Tạo URL để truy cập file đầu ra
    output_url = f"/temp/{os.path.basename(output_path)}"

    # Trả về response dạng stream với từng frame
    return Response(process_and_stream_video(input_path, output_path, task_id, output_url),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Route để theo dõi trạng thái task
@app.route('/status-stream/<task_id>')
def status_stream(task_id):
    # Kiểm tra nếu task_id không tồn tại
    if task_id not in processing_tasks:
        return jsonify({"error": "Task not found"}), 404  # Trả về lỗi 404

    # Hàm tạo event stream để gửi trạng thái
    def event_stream():
        while True:
            if task_id not in processing_tasks:
                break
            # Khóa để truy cập trạng thái task
            with processing_tasks[task_id]['lock']:
                if task_id not in processing_tasks:
                    break
                task = processing_tasks[task_id]  # Lấy thông tin task
                # Xác định trạng thái (ưu tiên paused nếu có)
                status = 'paused' if task['paused'] else task['status']
                # Tạo dữ liệu trạng thái
                status_data = {'status': status, 'url': task['url']}
                # Gửi dữ liệu dưới dạng Server-Sent Events
                yield f"data: {json.dumps(status_data)}\n\n"
                # Thoát nếu task hoàn tất hoặc lỗi
                if task['status'] in ['completed', 'error']:
                    break
            time.sleep(1)  # Đợi 1 giây trước khi gửi cập nhật tiếp theo
    # Trả về response dạng event stream
    return Response(event_stream(), mimetype="text/event-stream")

# Route để tạm dừng/tiếp tục stream
@app.route('/toggle-pause/<task_id>', methods=['POST'])
def toggle_pause(task_id):
    # Kiểm tra nếu task_id không tồn tại
    if task_id not in processing_tasks:
        return jsonify({"error": "Task not found"}), 404  # Trả về lỗi 404

    # Khóa để thay đổi trạng thái paused
    with processing_tasks[task_id]['lock']:
        if task_id not in processing_tasks:
            return jsonify({"error": "Task not found"}), 404  # Trả về lỗi 404
        # Kiểm tra nếu task không ở trạng thái streaming
        if processing_tasks[task_id]['status'] != 'streaming':
            return jsonify({"error": "Task is not streaming"}), 400  # Trả về lỗi 400
        # Đổi trạng thái paused (True -> False hoặc ngược lại)
        processing_tasks[task_id]['paused'] = not processing_tasks[task_id]['paused']
        # Xác định trạng thái mới
        new_state = "paused" if processing_tasks[task_id]['paused'] else "resumed"
        # Trả về trạng thái mới
        return jsonify({"status": new_state})

# Route để phục vụ file video đầu ra
@app.route('/temp/<path:filename>')
def serve_video(filename):
    # Đường dẫn tới file trong TEMP_DIR
    file_path = os.path.join(TEMP_DIR, filename)
    # Kiểm tra nếu file tồn tại
    if os.path.exists(file_path):
        # Trả về file video với mimetype video/mp4
        return send_from_directory(TEMP_DIR, filename, mimetype='video/mp4')
    # Trả về lỗi nếu file không tồn tại
    return jsonify({"error": "File not found"}), 404

# Kiểm tra nếu script chạy trực tiếp (không phải import)
if __name__ == '__main__':
    host = '127.0.0.1'  # Địa chỉ host (localhost)
    port = 5000         # Cổng chạy ứng dụng
    url = f'http://{host}:{port}'  # URL đầy đủ
    # Mở trình duyệt nếu không phải chạy lại từ Werkzeug
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        webbrowser.open(url)  # Tự động mở trình duyệt tới URL
    # Chạy ứng dụng Flask với debug và đa luồng
    app.run(debug=True, host=host, port=port, threaded=True)