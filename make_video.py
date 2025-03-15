import cv2
import os
import glob

# Đường dẫn tới thư mục chứa các file JPG
image_folder = 'D:/Github_Tin/VisDrone/video/sequences/uav0000119_02301_v_Copy'
# Đường dẫn tới file video đầu ra
output_video = 'D:/Github_Tin/VisDrone/video/sequences/uav0000119_02301_v_Copy/output.mp4'

# Lấy danh sách file JPG theo thứ tự
images = sorted(glob.glob(os.path.join(image_folder, '*.jpg')))

# Đọc ảnh đầu tiên để lấy kích thước khung hình
frame = cv2.imread(images[0])
height, width, layers = frame.shape

# Khởi tạo VideoWriter với codec MP4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec MP4
fps = 10  # Số khung hình trên giây (có thể thay đổi)
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Ghi từng ảnh vào video
for image in images:
    frame = cv2.imread(image)
    video_writer.write(frame)

# Giải phóng tài nguyên
video_writer.release()
print(f"Video đã được tạo: {output_video}")