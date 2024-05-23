#@title simswap gradio, works but not for gallery, delete pic_a path and pic_b path from testoptions py

"""
Author: Naiyuan liu
Github: https://github.com/NNNNAI
Date: 2021-11-23 17:03:58
LastEditors: Naiyuan liu
LastEditTime: 2021-11-24 19:19:43
Description: 
"""

import cv2
import torch
import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from insightface_func.face_detect_crop_single import Face_detect_crop
from util.reverse2original import reverse2wholeimage
import os
from util.add_watermark import watermark_image
from util.norm import SpecificNorm
from parsing_model.model import BiSeNet
from datetime import datetime
from gradio import gradio as gr

######## IP FACE ########to be added ##
import shutil

from insightface.app import FaceAnalysis
from insightface.utils import face_align
import torch
import os
from datetime import datetime
from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    AutoencoderKL,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
)
from PIL import Image
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlus
from diffusers.utils import load_image
import numpy as np

# Get the current date and time
now = datetime.now()
date_time = now.strftime("%Y-%m-%d_%H-%M-%S")









def generate_swap(face_image, image_path):
    def load_image(path):
        return Image.open(path)

    def process_file(fileobj):
        path = image_path + os.path.basename(fileobj)
        shutil.copyfile(fileobj.name, path)

    def lcm(a, b):
        return abs(a * b) / fractions.gcd(a, b) if a and b else 0

    transformer_Arcface = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    def _totensor(array):
        tensor = torch.from_numpy(array)
        img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255)

    opt = TestOptions().parse()
    start_epoch, epoch_iter = 1, 0
    crop_size = opt.crop_size

    torch.nn.Module.dump_patches = True
    if crop_size == 512:
        opt.which_epoch = 550000
        opt.name = "512"
        mode = "ffhq"
    else:
        mode = "None"

    logoclass = watermark_image("./simswaplogo/simswaplogo.png")
    model = create_model(opt)
    model.eval()

    spNorm = SpecificNorm()
    app = Face_detect_crop(
        name="antelope", root="/content/SimSwap/insightface_func/models"
    )
    app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640), mode=mode)

    with torch.no_grad():
        img_a_whole = np.array(face_image)
        img_a_align_crop, _ = app.get(img_a_whole, crop_size)
        img_a_align_crop_pil = Image.fromarray(
            cv2.cvtColor(img_a_align_crop[0], cv2.COLOR_BGR2RGB)
        )
        img_a = transformer_Arcface(img_a_align_crop_pil)
        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

        # convert numpy to tensor
        img_id = img_id.cuda()

        # create latent id
        img_id_downsample = F.interpolate(img_id, size=(112, 112))
        latend_id = model.netArc(img_id_downsample)
        latend_id = F.normalize(latend_id, p=2, dim=1)

        ############## Forward Pass ######################

        img_b_whole = cv2.imread(image_path)
        img_b_align_crop_list, b_mat_list = app.get(img_b_whole, crop_size)
        swap_result_list = []
        b_align_crop_tenor_list = []

        for b_align_crop in img_b_align_crop_list:
            b_align_crop_tenor = _totensor(
                cv2.cvtColor(b_align_crop, cv2.COLOR_BGR2RGB)
            )[None, ...].cuda()

            swap_result = model(
                None, b_align_crop_tenor, latend_id, None, True
            )[0]
            swap_result_list.append(swap_result)
            b_align_crop_tenor_list.append(b_align_crop_tenor)

        if opt.use_mask:
            n_classes = 19
            net = BiSeNet(n_classes=n_classes)
            net.cuda()
            save_pth = os.path.join(
                "/content/SimSwap/parsing_model", "79999_iter.pth"
            )
            net.load_state_dict(torch.load(save_pth))
            net.eval()
        else:
            net = None

        date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{date_time}.png"
        reverse2wholeimage(
            b_align_crop_tenor_list,
            swap_result_list,
            b_mat_list,
            crop_size,
            img_b_whole,
            logoclass,
            os.path.join(opt.output_path, filename),
            opt.no_simswaplogo,
            pasring_model=net,
            use_mask=opt.use_mask,
            norm=spNorm,
        )
        torch.cuda.empty_cache()
        
        return images, filename
with gr.Blocks()as interface:
  with gr.Tab("simswap"):
      with gr.Row():
          with gr.Column():
              face_image = gr.Image(label="Face Reference Image")
              image_path = gr.Textbox(label="Image Path")
              gallery = gr.Gallery(label="Swapped")
              generate_swap_btn = gr.Button("Swap")
              generate_swap_btn.click(
                  fn=generate_swap,
                  inputs=[face_image, image_path],
                  outputs=gallery,
              )

interface.launch(share=True)
