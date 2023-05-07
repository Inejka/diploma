import base64
import logging
from collections import UserDict
from io import BytesIO
from training_module import *
from gallery_builder import *
import gradio as gr
import requests

# ENABLE LOGGING--------------------------------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ],
    force=True
)
# ----------------------------------------------------------------------------------------------------------------------

# CREATE DATABASE------------------------------------------------------------------------------------------------------
if not os.path.exists("flagged"):
    os.mkdir("flagged")
conn = sqlite3.connect(os.path.join("flagged", "data.db"))
c = conn.cursor()
c.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='images' ''')
if not c.fetchone()[0] == 1:
    c.execute('''CREATE TABLE images
             (id INTEGER PRIMARY KEY, image_path TEXT, rating INTEGER)''')
    conn.commit()
    conn.close()
# ----------------------------------------------------------------------------------------------------------------------

# dowlonad model if it is missing----------------------------------------------------------------------------------------

if not os.path.exists("model.bin"):
    logging.info("Missing model, downloading")
    url = 'https://drive.google.com/uc?id=1uyIYjcPRg6TwIrLpa_p9bmCALtAax79F'
    output = 'model.bin'
    gdown.download(url, output, quiet=False)
    logging.info("Model downloaded")
# ----------------------------------------------------------------------------------------------------------------------

# DEFINE MODEl----------------------------------------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = get_model_definition()
model.load_state_dict(torch.load("model.bin"))

IMAGE_SIZE = 512
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
my_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD)
])
model = model.to(device)
for param in model.parameters():
    param.requires_grad = False

model.eval()


# ----------------------------------------------------------------------------------------------------------------------

# CREATE AND DIFINE BUFFE-----------------------------------------------------------------------------------------------
class Buffer(UserDict):
    def __init__(self, max_size=10, *args, **kwargs):
        self.max_size = max_size
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        if len(self) >= self.max_size:
            oldest_key = next(iter(self))
            del self[oldest_key]
        super().__setitem__(key, value)


buffer = Buffer(1000)
# ----------------------------------------------------------------------------------------------------------------------

headers = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36"}
with gr.Blocks() as app:
    gr.Markdown("Image scorer backend.")
    with gr.Tab("I am here for api only, pls don't touch me", visible=True):
        # this is for image adding for database
        img_input = gr.Textbox()
        score_input = gr.Textbox()
        button = gr.Button('Click me!')


        def add_image(inp, score=None):
            logging.info(f"Got image with src {inp} and score {score}")
            conn = sqlite3.connect(os.path.join("flagged", "data.db"))
            c = conn.cursor()
            c.execute("SELECT MAX(id) FROM images")
            result = c.fetchone()

            if result[0] is not None:
                next_id = result[0] + 1
            else:
                next_id = 1

            req = requests.get(inp, headers=headers)
            img = Image.open(BytesIO(req.content))
            img.save(os.path.join("flagged", str(next_id) + ".png"))
            c.execute("INSERT INTO images (image_path, rating) VALUES (?, ?)",
                      (os.path.join("flagged", str(next_id) + ".png"), score))
            conn.commit()
            conn.close()
            logging.info("Image added")
            return "readed"


        button.click(add_image, inputs=[img_input, score_input])

        # this is for connection test
        txt_output = gr.Textbox()
        button = gr.Button('Connection test')


        def return_ok():
            logging.info("Connected")
            return "connected"


        button.click(return_ok, outputs=txt_output)

        # this is for model predict
        img_input_predict = gr.Textbox()
        score_output = gr.Textbox()
        button = gr.Button('Predict')


        def predict(inp):
            if not str(inp):
                logging.info("Got empty string, returning -1")
                return "-1"
            if inp in buffer:
                logging.info(f"Using buffer, returned {buffer[inp]}")
                return buffer[inp]
            try:
                logging.info(f"Got: {inp}")
                if str(inp).find("http") != -1:
                    logging.info("Requesting image")
                    req = requests.get(inp, headers=headers)
                    logging.info("Got image")
                    img = Image.open(BytesIO(req.content)).convert('RGB')
                if str(inp).find("base64") != -1:
                    logging.info("Got base64, decoding")
                    image_bytes = base64.b64decode(inp.split(',')[1])
                    img = Image.open(BytesIO(image_bytes)).convert('RGB')
                logging.info("Transforming image")
                img = my_transforms(img).to(device).unsqueeze(0)
                logging.info("Predicting")
                ans = str(torch.argmax(model(img).sigmoid()).item() + 1)
                logging.info(f"Responsed with: {ans}")
                buffer[inp] = ans
                torch.cuda.empty_cache()
                return ans
            except Exception:
                logging.error("Error: " + str(Exception), exc_info=True)
                logging.error(f"Response status was {req.status_code}")
                return "-1"


        button.click(predict, inputs=img_input_predict, outputs=score_output)

    with gr.Tab("Train tab", visible=True):
        with gr.Row():
            epochs = gr.Number(value=10, label="Epochs", interactive=True)
            batch_size = gr.Number(value=9, label="Batch size", interactive=True)
        with gr.Row():
            l4lr = gr.Number(value=1e-5, label="layer4.parameter LR", interactive=True)
            fclr = gr.Number(value=5e-3, label="fc.parameter LR", interactive=True)
            adamLR = gr.Number(value=1e-3, label="adam LR", interactive=True)
        button = gr.Button('Train model from ram')


        def standart_eval_prep():
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
            torch.cuda.empty_cache()
            buffer.clear()


        def train_with_flagged(lr1, lr2, lr3, batch_size_pm, epochs_pm):
            try:
                logging.info("Starting training")
                global model
                model = train_model_from_ram_with_flagged(model, my_transforms, int(epochs_pm), int(batch_size_pm),
                                                          lrs=[lr1, lr2, lr3])
                standart_eval_prep()
                logging.info("Training finished")
            except:
                logging.error(str(Exception), exc_info=True)
                standart_eval_prep()


        button.click(train_with_flagged, inputs=[l4lr, fclr, adamLR, batch_size, epochs])

        button = gr.Button('Train model from pretrained')


        def train_from_pretrained(lr1, lr2, lr3, batch_size_pm, epochs_pm):
            try:
                logging.info("Starting training")
                global model
                model = train_model_from_my_pretrained(my_transforms, int(epochs_pm), int(batch_size_pm),
                                                       lrs=[lr1, lr2, lr3])
                standart_eval_prep()
                logging.info("Training finished")
            except Exception:
                logging.error(str(Exception), exc_info=True)
                standart_eval_prep()


        button.click(train_from_pretrained, inputs=[l4lr, fclr, adamLR, batch_size, epochs])

        button = gr.Button("Load standart pretrained")


        def load_standart_pretrained():
            logging.info("Loading model")
            global model
            model.load_state_dict(torch.load("model.bin"))
            standart_eval_prep()
            logging.info("Model loaded")


        button.click(load_standart_pretrained)
    build_gallery("Database tab", NUM_ROWS, NUM_COLUMNS)
