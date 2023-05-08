import logging
import os.path
import shutil

import gradio as gr
import torch
from PIL import Image
from icrawler.builtin import GoogleImageCrawler

NUM_ROWS_S = 2
NUM_COLUMNS_S = 4

model_suggestion = None
transforms_suggestion = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_model(model):
    global model_suggestion
    model_suggestion = model


def set_transforms(transforms):
    global transforms_suggestion
    transforms_suggestion = transforms


def get_suggestions(text, type_t, color, size, license_t, to_parse, score_slice, use_filter):
    global model_suggestion
    global transforms_suggestion
    to_return = []
    if not os.path.exists("temp"):
        os.mkdir("temp")
    score_slice = int(score_slice)

    google_crawler = GoogleImageCrawler(storage={'root_dir': 'temp'}, feeder_threads=1,
                                        parser_threads=1,
                                        downloader_threads=4)
    filters = dict(
        size=size,
        color=color,
        license=license_t,
        type=type_t)
    google_crawler.crawl(keyword=text, max_num=int(to_parse), filters=filters if use_filter else None)

    for image_path in os.listdir("temp"):
        try:
            logging.info(f"Predicting {os.path.join('temp', image_path)}")
            temp_img = Image.open(os.path.join("temp", image_path)).convert('RGB')
            img = transforms_suggestion(temp_img).to(device).unsqueeze(0)
            ans = (torch.argmax(model_suggestion(img).sigmoid()).item() + 1)
            torch.cuda.empty_cache()
            logging.info(f"Got score: {ans}")
            if ans >= score_slice:
                to_return.append(temp_img)
        except Exception:
            logging.error(str(Exception))
        if len(to_return) >= NUM_ROWS_S * NUM_COLUMNS_S:
            break
    while len(to_return) < NUM_ROWS_S * NUM_COLUMNS_S:
        to_return.append(None)
    shutil.rmtree("temp")
    return to_return


def build_suggestion(name, rows, columns):
    to_update = []
    with gr.Tab(name) as tab:
        with gr.Column():
            text_input = gr.Textbox()
            with gr.Row():
                type_dp = gr.Dropdown(["photo", "face", "clipart", "linedrawing", "animated"], value="photo",
                                      label="Type")
                color_dp = gr.Dropdown(
                    ["color", "blackandwhite", "transparent", "red", "orange", "yellow", "green", "teal", "blue",
                     "purple", "pink", "white", "gray", "black", "brown"], value="color", label="Color")
                size = gr.Textbox(label="Size", value="medium")
                license_dp = gr.Dropdown(["noncommercial", "commercial", "noncommercial,modify", "commercial,modify"],
                                         value="noncommercial", label="License")
                to_parse = gr.Number(label="To Parse images", value=30)
                score_slice = gr.Dropdown(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"], value="8",
                                          label="Score threshold")
                use_filter= gr.Checkbox(value=True, label="Use filter")
            find_button = gr.Button("Get Suggestions")

        for i in range(rows):
            with gr.Row():
                for j in range(columns):
                    with gr.Column():
                        image = gr.Image(interactive=False)
                        to_update.append(image)

        find_button.click(get_suggestions,
                          inputs=[text_input, type_dp, color_dp, size, license_dp, to_parse, score_slice, use_filter],
                          outputs=to_update)
