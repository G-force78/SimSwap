#@title simswap and IP adapter for use in colab notebook

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
import tempfile

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



with tempfile.TemporaryDirectory() as temp_dir:
  # Set the temporary directory for Gradio within the block
  os.environ["GRADIO_TEMP_DIR"] = temp_dir

  # Your Gradio interface code here


#### IP ADAPTER #####

def generate_image(
    prompt,
    negative_prompt,
    depth_map_dir,
    face_reference_image,
    s_scale,
    strength,
    guidance_scale,
    num_inference_steps,
    seed,
    v2,
):
    # Get the current date and time
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Create the output directory if it doesn't exist
    output_dir = "/content/SimSwap/output"
    os.makedirs(output_dir, exist_ok=True)
    # depth_map_dir = "" # or whichever you have the depthmap images in

    app = FaceAnalysis(
        name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    face_reference_image = face_reference_image  # the face reference image
    face_reference_image_np = np.array(face_reference_image)
    faces = app.get(face_reference_image_np)
    faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
    face_image = face_align.norm_crop(
        face_reference_image_np, landmark=faces[0].kps, image_size=224
    )  # you can also segment the face

    # base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
    base_model_path = "emilianJR/epiCRealism"
    vae_model_path = "stabilityai/sd-vae-ft-mse"
    image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    ip_ckpt = (
        "/content/ip-adapter-faceid-plus_sd15.bin"
        if not v2
        else "ip-adapter-faceid-plusv2_sd15.bin"
    )
    device = "cuda"

    # Control net test
    # controlnet_model_path = "/content/checkpoints" #change to local path
    # controlnet_model_path = "lllyasviel/control_v11p_sd15_openpose"
    controlnet_model_path = "lllyasviel/control_v11f1p_sd15_depth"
    controlnet = ControlNetModel.from_pretrained(
        controlnet_model_path, torch_dtype=torch.float16
    )

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )

    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        controlnet=controlnet,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None,
    )

    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()

    # load ip-adapter
    ip_model = IPAdapterFaceIDPlus(pipe, image_encoder_path, ip_ckpt, device)

    depth_map_files = [
        f for f in os.listdir(depth_map_dir) if f.endswith((".jpg", ".png"))
    ]
    images = []

    for idx, filename in enumerate(depth_map_files):
        depth_map_path = os.path.join(depth_map_dir, filename)
        depth_map = load_image(depth_map_path)

        image = ip_model.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=depth_map,
            face_image=face_image,
            faceid_embeds=faceid_embeds,
            shortcut=v2,
            s_scale=s_scale,
            strength=strength,
            guidance_scale=guidance_scale,
            num_samples=1,  # Generate one image per depth map
            width=512,
            height=512,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )[0]

        # Save the image with the prompt name, date/time, and depth map index
        image_name = f"{prompt.replace(' ', '_')}_{date_time}_{idx}_0.png"
        image_path = os.path.join(output_dir, image_name)
        image.save(image_path)
        images.append(image)

    torch.cuda.empty_cache()
    return images



#### SIMSWAP #######

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

        images = []
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
        # Read the image
        swapped_image_path = os.path.join(opt.output_path, filename)
        swapped_image = Image.open(swapped_image_path)
        return swapped_image
        
#### GRADIO UI ####        
        
with gr.Blocks()as interface:
  with gr.Tab("simswap"):
      with gr.Row():
          with gr.Column():
              face_image = gr.Image(label="Face Reference Image")
              image_path = gr.Textbox(label="Image Path")
              swapped_image = gr.Image(label="Swapped Image")
              generate_swap_btn = gr.Button("Swap")
              generate_swap_btn.click(
                  fn=generate_swap,
                  inputs=[face_image, image_path],
                  outputs=swapped_image,
              )
  with gr.Tab("IP adapter"):
    with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt")
                negative_prompt = gr.Textbox(label="Negative Prompt")
                depth_map_dir = gr.Textbox(label="Depth Map Directory")
                face_reference_image = gr.Image(
                    label="Face Reference Image", type="pil"
                )
                # s_scale = gr.Slider(label="Face Structure strength", value=0.6, step=0.1, minimum=0, maximum=3)
                # num_inference_steps = gr.Slider(label="steps", value=10, step=1, minimum=1, maximum=50)
                v2 = gr.Checkbox(label="Use v2 Adapter", value=False)

            with gr.Column():
                s_scale = gr.Slider(
                    label="Face Structure strength",
                    value=0.6,
                    step=0.1,
                    minimum=0,
                    maximum=3,
                )
                num_inference_steps = gr.Slider(
                    label="steps", value=10, step=1, minimum=1, maximum=50
                )
                strength = gr.Slider(
                    label="Control Strength",
                    minimum=0.0,
                    maximum=2.0,
                    value=1.0,
                    step=0.01,
                )
                guidance_scale = gr.Slider(
                    label="Guidance Scale",
                    minimum=0.1,
                    maximum=30.0,
                    value=9.0,
                    step=0.1,
                )
                seed = gr.Slider(
                    label="seed", minimum=10, maximum=5000, value=50, step=1
                )
                gallery = gr.Gallery(label="Generated Images")

            generate_btn = gr.Button("Generate Images")
            generate_btn.click(
            fn=generate_image,
            inputs=[
                prompt,
                negative_prompt,
                depth_map_dir,
                face_reference_image,
                s_scale,
                strength,
                guidance_scale,
                num_inference_steps,
                seed,
                v2,
            ],
            outputs=gallery,
        )



interface.launch(share=True)
