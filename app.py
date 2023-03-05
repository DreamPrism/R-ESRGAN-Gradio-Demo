import psutil
from subprocess import call
import sys
from random import randint
import torch
from PIL import Image
import gradio as gr
import random
import os


def run_cmd(command):
    try:
        print(command)
        call(command, shell=True)
    except KeyboardInterrupt:
        print("Process interrupted")
        sys.exit(1)


def inference(img, mode):
    _id = randint(1, 10000)
    INPUT_DIR = "/tmp/input_image" + str(_id) + "/"
    OUTPUT_DIR = "/tmp/output_image" + str(_id) + "/"
    run_cmd("rm -rf " + INPUT_DIR)
    run_cmd("rm -rf " + OUTPUT_DIR)
    run_cmd("mkdir " + INPUT_DIR)
    run_cmd("mkdir " + OUTPUT_DIR)
    img.save(INPUT_DIR + "1.png", "PNG")
    if mode == "base":
        run_cmd("python inference_realesrgan.py -n RealESRGAN_x4plus -i " + INPUT_DIR + " -o " + OUTPUT_DIR)
    else:
        run_cmd("python inference_realesrgan.py -n RealESRGAN_x4plus_anime_6B -i " + INPUT_DIR + " -o " + OUTPUT_DIR)
    return os.path.join(OUTPUT_DIR, "1_out.png")


title = "Real-ESRGAN"
description = "<center>Real-ESRGAN超分辨率模型的Gradio Demo<br>一次请提交一张图片<br>动漫插图等图片请选择anime模式<br>其他图片请选择base模式</center>"
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2107.10833'>Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data</a> | <a href='https://github.com/xinntao/Real-ESRGAN'>Github Repo</a></p>"

gr.Interface(
    inference,
    [gr.inputs.Image(type="pil", label="Input"), gr.inputs.Radio(
        ["base", "anime"], type="value", default="anime", label="模式")],
    gr.outputs.Image(type="file", label="Output"),
    title=title,
    description=description,
    article=article).launch(share=True)
