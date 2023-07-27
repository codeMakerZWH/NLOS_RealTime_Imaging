import os
import torch
from cv2 import cv2
from torchvision.transforms import functional as F
from data_load_2 import test_dataloader
from utils import Adder

from skimage.metrics import peak_signal_noise_ratio
from torchvision.transforms import functional as F
import time

def _eval(model, args):
    state_dict = torch.load(args.test_model)
    model.load_state_dict(state_dict['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = test_dataloader(args.data_dir,args.data_dir1,args.data_dir2, batch_size=1, num_workers=0)
    torch.cuda.empty_cache()
    adder = Adder()
    model.eval()
    with torch.no_grad():
        psnr_adder = Adder()

        # Hardware warm-up
        for iter_idx, data in enumerate(dataloader):
            input_img, label_img, input_img1, label_img1,input_img2,name = data
            input_img = input_img.to(device)
            input_img1 = input_img1.to(device)
            input_img2 = input_img2.to(device)
            tm = time.time()
            _ = model(input_img,input_img2,input_img1)
            _ = time.time() - tm

            if iter_idx == 20:
                break

        # Main Evaluation
        for iter_idx, data in enumerate(dataloader):
            input_img, label_img, input_img1, label_img1, input_img2, name = data
            # input_img, label_img,name = data
            name = name[0]
            input_img = input_img.to(device)
            input_img1 = input_img1.to(device)
            input_img2 = input_img2.to(device)

            tm = time.time()

            pred = model(input_img,input_img2,input_img1)

            elapsed = time.time() - tm
            adder(elapsed)


            pred_clip = torch.clamp(pred, 0, 1)
            p_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()
            psnr = peak_signal_noise_ratio(p_numpy, label_numpy, data_range=1)



            if args.save_image:
                save_name = os.path.join(args.result_dir,name[0]+'.bmp')