import gradio as gr
import os
import cv2
import numpy as np
from moviepy.editor import *
from share_btn import community_icon_html, loading_icon_html, share_js

from diffusers import StableDiffusionInstructPix2PixPipeline
import torch
from PIL import Image, ImageOps
import time
import psutil
import math
import random


pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=torch.float16, safety_checker=None)

device = "GPU 🔥" if torch.cuda.is_available() else "CPU 🥶"

if torch.cuda.is_available():
    pipe = pipe.to("cuda")


def pix2pix(
        input_image: Image.Image,
        instruction: str,
        steps: int,
        seed: int,
        text_cfg_scale: float,
        image_cfg_scale: float,
    ):
        
        width, height = input_image.size
        factor = 512 / max(width, height)
        factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
        width = int((width * factor) // 64) * 64
        height = int((height * factor) // 64) * 64
        input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)

        if instruction == "":
            return [input_image, seed]

        generator = torch.manual_seed(seed)
        edited_image = pipe(
            instruction, image=input_image,
            guidance_scale=text_cfg_scale, image_guidance_scale=image_cfg_scale,
            num_inference_steps=steps, generator=generator,
        ).images[0]
        print(f"EDITED: {edited_image}")
        return edited_image



def get_frames(video_in):
    frames = []
    #resize the video
    clip = VideoFileClip(video_in)
    
    #check fps
    if clip.fps > 30:
        print("vide rate is over 30, resetting to 30")
        clip_resized = clip.resize(height=512)
        clip_resized.write_videofile("video_resized.mp4", fps=30)
    else:
        print("video rate is OK")
        clip_resized = clip.resize(height=512)
        clip_resized.write_videofile("video_resized.mp4", fps=clip.fps)
    
    print("video resized to 512 height")
    
    # Opens the Video file with CV2
    cap= cv2.VideoCapture("video_resized.mp4")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("video fps: " + str(fps))
    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite('kang'+str(i)+'.jpg',frame)
        frames.append('kang'+str(i)+'.jpg')
        i+=1
    
    cap.release()
    cv2.destroyAllWindows()
    print("broke the video into frames")
    
    return frames, fps


def create_video(frames, fps):
    print("building video result")
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile("movie.mp4", fps=fps)
    
    return 'movie.mp4'


def infer(prompt,video_in, seed_in, trim_value):
    print(prompt)
    break_vid = get_frames(video_in)
    
    frames_list= break_vid[0]
    fps = break_vid[1]
    n_frame = int(trim_value*fps)
    
    if n_frame >= len(frames_list):
        print("video is shorter than the cut value")
        n_frame = len(frames_list)
    
    result_frames = []
    print("set stop frames to: " + str(n_frame))
    
    for i in frames_list[0:int(n_frame)]:
        pil_i = Image.open(i).convert("RGB")
        
        pix2pix_img = pix2pix(pil_i, prompt, 50, seed_in, 7.5, 1.5)
        #print(pix2pix_img)
        #image = Image.open(pix2pix_img)
        #rgb_im = image.convert("RGB")
  
        # exporting the image
        pix2pix_img.save(f"result_img-{i}.jpg")
        result_frames.append(f"result_img-{i}.jpg")
        print("frame " + i + "/" + str(n_frame) + ": done;")

    final_vid = create_video(result_frames, fps)
    print("finished !")
    
    return final_vid, gr.Group.update(visible=True)

title = """
    <div style="text-align: center; max-width: 700px; margin: 0 auto;">
        <div
        style="
            display: inline-flex;
            align-items: center;
            gap: 0.8rem;
            font-size: 1.75rem;
        "
        >
        <h1 style="font-weight: 900; margin-bottom: 7px; margin-top: 5px;">
            Pix2Pix Video
        </h1>
        </div>
        <p style="margin-bottom: 10px; font-size: 94%">
        Apply Instruct Pix2Pix Diffusion to a video 
        </p>
    </div>
"""

article = """
    
    <div class="footer">
        <p>
        Examples by <a href="https://twitter.com/CitizenPlain" target="_blank">Nathan Shipley</a> •&nbsp;
        Follow <a href="https://twitter.com/fffiloni" target="_blank">Sylvain Filoni</a> for future updates 🤗
        </p>
    </div>
    <div id="may-like-container" style="display: flex;justify-content: center;flex-direction: column;align-items: center;margin-bottom: 30px;">
        <p>You may also like: </p>
        <div id="may-like-content" style="display:flex;flex-wrap: wrap;align-items:center;height:20px;">
            
            <svg height="20" width="162" style="margin-left:4px;margin-bottom: 6px;">       
                 <a href="https://huggingface.co/spaces/timbrooks/instruct-pix2pix" target="_blank">
                    <image href="https://img.shields.io/badge/🤗 Spaces-Instruct_Pix2Pix-blue" src="https://img.shields.io/badge/🤗 Spaces-Instruct_Pix2Pix-blue.png" height="20"/>
                 </a>
            </svg>
            
        </div>
    
    </div>
    
"""

with gr.Blocks(css='style.css') as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML(title)
        with gr.Row():
            with gr.Column():
                video_inp = gr.Video(label="Video source", source="upload", type="filepath", elem_id="input-vid")
                prompt = gr.Textbox(label="Prompt", placeholder="enter prompt", show_label=False, elem_id="prompt-in")
                with gr.Row():
                    seed_inp = gr.Slider(label="Seed", minimum=0, maximum=2147483647, step=1, value=123456)
                    trim_in = gr.Slider(label="Cut video at (s)", minimun=1, maximum=60, step=1, value=1)
            with gr.Column():
                video_out = gr.Video(label="Pix2pix video result", elem_id="video-output")
                gr.HTML("""
                <a style="display:inline-block" href="https://huggingface.co/spaces/fffiloni/Pix2Pix-Video?duplicate=true"><img src="https://img.shields.io/badge/-Duplicate%20Space-blue?labelColor=white&style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAP5JREFUOE+lk7FqAkEURY+ltunEgFXS2sZGIbXfEPdLlnxJyDdYB62sbbUKpLbVNhyYFzbrrA74YJlh9r079973psed0cvUD4A+4HoCjsA85X0Dfn/RBLBgBDxnQPfAEJgBY+A9gALA4tcbamSzS4xq4FOQAJgCDwV2CPKV8tZAJcAjMMkUe1vX+U+SMhfAJEHasQIWmXNN3abzDwHUrgcRGmYcgKe0bxrblHEB4E/pndMazNpSZGcsZdBlYJcEL9Afo75molJyM2FxmPgmgPqlWNLGfwZGG6UiyEvLzHYDmoPkDDiNm9JR9uboiONcBXrpY1qmgs21x1QwyZcpvxt9NS09PlsPAAAAAElFTkSuQmCC&logoWidth=14" alt="Duplicate Space"></a> 
                work with longer videos / skip the queue: 
                """, elem_id="duplicate-container")
                submit_btn = gr.Button("Generate Pix2Pix video")

                with gr.Group(elem_id="share-btn-container", visible=False) as share_group:
                    community_icon = gr.HTML(community_icon_html)
                    loading_icon = gr.HTML(loading_icon_html)
                    share_button = gr.Button("Share to community", elem_id="share-btn")
        
        inputs = [prompt,video_inp,seed_inp, trim_in]
        outputs = [video_out, share_group]
        
        #ex = gr.Examples(
        #    [
        #        ["Make it a marble sculpture", "./examples/pexels-jill-burrow-7665249_512x512.mp4", 422112651, 4],
        #        ["Make it molten lava", "./examples/Ocean_Pexels_ 8953474_512x512.mp4", 43571876, 4]
        #    ],
        #    inputs=inputs,
        #    outputs=outputs,
        #    fn=infer,
        #    cache_examples=True,
        #)
        
        gr.HTML(article)
    
    submit_btn.click(infer, inputs, outputs)
    share_button.click(None, [], [], _js=share_js)

    
    
demo.queue(max_size=12).launch(share=True)
