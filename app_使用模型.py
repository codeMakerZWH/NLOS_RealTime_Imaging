import cv2
import numpy as np
import time
from threading import Thread, Lock
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms import functional as Fs
from TRO_Net import TRO_Net
import torch
from flask import Flask, render_template, Response

app = Flask(__name__)


model = TRO_Net()
state_dict = torch.load("modeltest.pkl")
# state_dict = torch.load("model.pkl")
model.load_state_dict(state_dict['model'])
model.eval()  # Set the model to evaluation mode
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


rtsp_url_vis = "rtsp://192.168.1.100/ch0/stream0"  ##可见光
rtsp_url_ir = "rtsp://192.168.1.100/ch1/stream0"  ##红外

cap_vis = cv2.VideoCapture(rtsp_url_vis)
cap_ir = cv2.VideoCapture(rtsp_url_ir)

desired_fps = 25
cap_vis.set(cv2.CAP_PROP_FPS, desired_fps)
cap_ir.set(cv2.CAP_PROP_FPS, desired_fps)

cap_vis.set(cv2.CAP_PROP_BUFFERSIZE, 10)
cap_ir.set(cv2.CAP_PROP_BUFFERSIZE, 10)

vis_frame = None
ir_frame = None
output_frame = None
frame_lock = Lock()


def bilateralFilter(img, d, sigmaColor, sigmaSpace):
    return cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)
#
def laplacian_filter(image):
    # 应用Laplacian滤波器
    filtered_image = cv2.Laplacian(image, cv2.CV_64F)
    filtered_image = np.uint8(np.absolute(filtered_image))
    return filtered_image

def sobel_filter(image, dx, dy, ksize):
    # 应用Sobel滤波器
    filtered_image = cv2.Sobel(image, cv2.CV_64F, dx, dy, ksize)
    filtered_image = cv2.convertScaleAbs(filtered_image)
    return filtered_image

def scharr_filter(image, dx, dy):
    # 应用Scharr滤波器
    filtered_image = cv2.Scharr(image, cv2.CV_64F, dx, dy)
    filtered_image = cv2.convertScaleAbs(filtered_image)
    return filtered_image

def gaussian_filter(image, kernel_size, sigma):
    # 应用高斯滤波器
    filtered_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return filtered_image

def median_filter(image, kernel_size):
    # 应用中值滤波器
    filtered_image = cv2.medianBlur(image, kernel_size)
    return filtered_image


import numpy as np

def grabcut_segmentation(image):
    # 创建一个与图像大小相同的掩膜
    mask = np.zeros(image.shape[:2], np.uint8)

    # 定义背景和前景模型
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)

    # 定义感兴趣区域（ROI）
    rectangle = (50, 50, image.shape[1]-50, image.shape[0]-50)

    # 执行GrabCut算法
    cv2.grabCut(image, mask, rectangle, background_model, foreground_model, 5, cv2.GC_INIT_WITH_RECT)

    # 创建一个掩膜，将确定的前景和可能的前景设置为1，其他区域设置为0
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # 将原始图像与掩膜相乘以突出人体轮廓
    segmented_image = image * mask2[:, :, np.newaxis]
    return segmented_image


def adaptive_histogram_equalization(image, clip_limit, tile_size):
    # 将图像转换为8位灰度图像
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 将图像转换为8位无符号整数
    image = cv2.convertScaleAbs(image)

    # 应用自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    equalized_image = clahe.apply(image)
    return equalized_image

def enhance_contours_canny(image, threshold1, threshold2):
    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用Canny边缘检测算法
    edges = cv2.Canny(gray_image, threshold1, threshold2)

    # 将边缘和原始图像融合，以加强轮廓
    filtered_img = cv2.bitwise_and(image, image, mask=edges)

    return filtered_img

def enhance_contours_with_fusion(image, threshold1, threshold2, alpha=0.8):
    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用Canny边缘检测算法
    edges = cv2.Canny(gray_image, threshold1, threshold2)

    # 将边缘图像转换为3通道（彩色图像）方便融合
    edges_3channel = cv2.merge((edges, edges, edges))

    # 融合边缘图像和原始图像
    fused_img = cv2.addWeighted(image, alpha, edges_3channel, 1 - alpha, 0)

    return fused_img

def enhance_contours_with_threshold(image, threshold_value, alpha=0.8):
    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用阈值分割提取人体轮廓
    _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

    # 将二值化图像转换为3通道（彩色图像）方便融合
    binary_image_3channel = cv2.merge((binary_image, binary_image, binary_image))

    # 融合二值化图像和原始图像
    fused_img = cv2.addWeighted(image, alpha, binary_image_3channel, 1 - alpha, 0)

    return fused_img

# def BilateralFilter

def f(ir,vis, alpha=0.8):
    # 融合二值化图像和原始图像
    return cv2.addWeighted(ir, alpha, vis, 1 - alpha, 0)


##原始备份
# def process_ir(ir):
#     ir = cv2.resize(ir, (640, 512))
#     ir = ir[:, 120:520]
#     ir1 = cv2.resize(ir, (256, 256))
#     ir1 = F.to_tensor(ir1)
#     ir1 = ir1.unsqueeze(dim=0)
#     return ir1,ir
#
# def process_vis(vis):
#     vis = cv2.resize(vis, (1280, 720))
#     # vis = vis[140:580, 160:760 + 160]
#     vis = vis[140 - 60:580, 160:760 + 230]  ##高 宽
#     vis = cv2.resize(vis, (640, 512))
#     vis = vis[:, 120:520]
#     vis = cv2.resize(vis, (256, 256))
#     vis1 = F.to_tensor(vis)
#     vis1 = vis1.unsqueeze(dim=0)
#     return vis1,vis

#融合图像
def process_ir(ir):
    ir = cv2.resize(ir, (640, 512))
    ir = ir[:, 120:520]
    ir1 = ir.copy()
    ir1 = cv2.resize(ir1, (256, 256))
    ir1 = F.to_tensor(ir1)
    ir1 = ir1.unsqueeze(dim=0)
    return ir1,ir

def process_vis(vis):
    vis = cv2.resize(vis, (1280, 720))
    vis = vis[140-50:580, 160:760 + 250] ##高 宽
    vis = cv2.resize(vis, (640, 512))
    vis = vis[:, 120:520]
    vis1 = vis.copy()
    vis = cv2.resize(vis1, (256, 256))
    # vis1 = F.to_tensor(vis)
    # vis1 = vis1.unsqueeze(dim=0)

    vis2 = cv2.resize(vis, (256, 256))
    vis2 = F.to_tensor(vis2)
    vis2 = vis2.unsqueeze(dim=0)

    return vis2,vis1





def process_and_display_output():
    global vis_frame, ir_frame, output_frame
    timestamp = None
    while True:
        with frame_lock:
            if vis_frame is None or ir_frame is None:
                continue

            vis = vis_frame.copy()
            ir = ir_frame.copy()
            if timestamp is None:
                timestamp = time.time()

        vis,vis1 = process_vis(vis)
        ir,ir1 = process_ir(ir)

        filterdImg = f(ir1, vis1, alpha=0.25)
        vis_out = cv2.resize(vis1,(750, 650))
        # diameter = 15  # 控制像素领域的直径
        # sigma_color = 75  # 控制颜色相似性的标准差
        # sigma_space = 75  # 控制空间相似性的标准差
        #
        # # 应用双边模糊
        # vis_out = cv2.bilateralFilter(vis_out, diameter, sigma_color, sigma_space)

        kernel_size = (3, 3)  # 设置高斯核大小，必须是奇数
        sigma_x = 0  # 在X方向上的标准差（如果为0，根据核的大小自动计算）
        sigma_y = 0  # 在Y方向上的标准差（如果为0，根据核的大小自动计算）

        # 应用高斯模糊
        vis_out = cv2.GaussianBlur(vis_out, kernel_size, sigma_x, sigma_y)

        filterdImg = cv2.resize(filterdImg, (750, 650))
        # Process vis1 and ir1 to create output
        with torch.no_grad():
            vis = vis.to(device)
            ir = ir.to(device)
            output = model(vis, ir, ir)
            pred_clip = torch.clamp(output, 0, 1)
            p_numpy = pred_clip.squeeze(0).cpu().numpy()
            p_numpy = np.transpose(p_numpy, (1, 2, 0))
            output_np = (p_numpy * 255).astype(np.uint16)
        # Resize output_np to the desired shape (512, 400, 3)

        ##vis+out
        output_np_resized = cv2.resize(output_np, (512, 512))
        vis1_np_resized = cv2.resize(vis1, (512, 512))
        # Combine the two processed frames
        output = np.concatenate((vis1_np_resized, output_np_resized), axis=1)

        #only out
        # output = cv2.resize(output_np, (720, 512))
        # output = cv2.resize(output_np, (512, 512))


        # img = cv2.resize(vis, (512, 512))
        # filterdImg = bilateralFilter(img, 9, 75, 75)#双边滤波算法
        # filterdImg = laplacian_filter(img)#拉普拉斯滤波算法
        # filterdImg = sobel_filter(img, 1, 1, 3)#sobel滤波算法
        # filterdImg = scharr_filter(img, 1, 0) #scharr滤波算法
        # filterdImg = adaptive_histogram_equalization(img, 2.0, 8) #自适应直方图均衡化
        # filterdImg = gaussian_filter(img, 5, 1.5) #高斯滤波
        # filterdImg = median_filter(img, 3) #中值滤波
        # filterdImg = grabcut_segmentation(filterdImg)  # grabcut分割算法
        # filterdImg = enhance_contours_canny(filterdImg,200,300)
        # filterdImg = enhance_contours_with_fusion(img, 50, 300, alpha=0.7)  # 调整阈值和alpha来控制效果
        # filterdImg = enhance_contours_with_threshold(img, 65, alpha=0.9)  # 调整alpha来控制融合的程度

        # output = cv2.resize(output_np, (512, 512))

        # vis1_np_resized = cv2.resize(vis1, (512, 512))
        # Combine the two processed frames
        # output = np.concatenate((filterdImg, vis_out), axis=1)
        # output = filterdImg



        # Combine the two processed frames


        with frame_lock:
            output_frame = output.copy()

        # Wait until the next frame is due
        timestamp += 1 / desired_fps
        time.sleep(max(0, timestamp - time.time()))

def capture_frames_vis():
    global vis_frame
    while True:
        ret_vis, vis = cap_vis.read()
        if not ret_vis:
            break
        with frame_lock:
            vis_frame = vis.copy()

def capture_frames_ir():
    global ir_frame
    while True:
        ret_ir, ir = cap_ir.read()
        if not ret_ir:
            break
        with frame_lock:
            ir_frame = ir.copy()



@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    while True:
        with frame_lock:
            if output_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', output_frame)
            frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

output_thread = Thread(target=app.run, args=('0.0.0.0',), kwargs={'debug': False, 'threaded': True})
output_thread.start()

capture_thread_vis = Thread(target=capture_frames_vis)
capture_thread_ir = Thread(target=capture_frames_ir)
process_and_display_thread = Thread(target=process_and_display_output)

capture_thread_vis.start()
capture_thread_ir.start()
process_and_display_thread.start()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Release the video capture objects and close any windows when the output thread is finished
output_thread.join()
cap_ir.release()
cap_vis.release()
cv2.destroyAllWindows()
