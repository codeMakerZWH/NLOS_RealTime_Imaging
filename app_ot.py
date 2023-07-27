import cv2
import numpy as np
import time
from threading import Thread, Lock
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms import functional as Fs
from introvae import IntroAEEncoder,IntroAEDecoder,IntroAE
import torch
from flask import Flask, render_template, Response
import torchvision.transforms as transforms
app = Flask(__name__)


str_to_list = lambda x: [int(xi) for xi in x.split(',')]
channels = '64, 128, 256, 512, 512, 512'
netG = IntroAE(norm="batch", gpuId = "0",cdim=3,channels=str_to_list(channels))
encoder =IntroAEEncoder(norm="batch",cdim=3,channels=str_to_list(channels))

en_state_dict = torch.load("28_net_G_Encoder2.pth")
de_state_dict = torch.load("28_net_G_Decoder.pth")

encoder.load_state_dict(en_state_dict)
netG.decoder.load_state_dict(de_state_dict)
encoder.eval()  # Set the model to evaluation mode
netG.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder.to(device)
netG.to(device)

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

def process_ir(ir):
    # ir = cv2.resize(ir, (640, 512))
    # ir = ir[:, 120:520]
    # ir1 = cv2.resize(ir, (256, 256))
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]
    transform = transforms.Compose(transform_list)
    ir1 = F.to_tensor(ir)
    ir1 = ir1.unsqueeze(dim=0)
    return ir1,ir

def process_vis(vis):
    # vis = cv2.resize(vis, (1280, 720))
    # vis = vis[140:580, 160:760 + 160]
    # vis = cv2.resize(vis, (640, 512))
    # vis = vis[:, 120:520]
    vis = cv2.resize(vis, (256, 256))
    vis1 = F.to_tensor(vis)
    vis1 = vis1.unsqueeze(dim=0)
    return vis1,vis


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

        # Process vis1 and ir1 to create output
        with torch.no_grad():
            vis = vis.to(device)
            ir = ir.to(device)
            z = encoder(ir)
            output = netG.decoder(z)

            pred_clip = torch.clamp(output, 0, 1)
            p_numpy = pred_clip.squeeze(0).cpu().numpy()
            p_numpy = np.transpose(p_numpy, (1, 2, 0))
            output_np = (p_numpy * 255).astype(np.uint8)
        # Resize output_np to the desired shape (512, 400, 3)

        ##vis+out
        output_np_resized = cv2.resize(output_np, (512, 512))
        vis1_np_resized = cv2.resize(vis1, (512, 512))
        # Combine the two processed frames
        output = np.concatenate((vis1_np_resized, output_np_resized), axis=1)

        #only out
        # output = cv2.resize(output_np, (720, 512))

        # Remove red channel from the output
        # output_np_resized[:, :, 2] = 0


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
