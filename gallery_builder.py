import os.path
import sqlite3

import gradio as gr
from PIL import Image

NUM_ROWS = 1
NUM_COLUMNS = 4


def rescore(page, image_pos, score):
    page = page - 1
    conn = sqlite3.connect(os.path.join("flagged", "data.db"))
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM images LIMIT ? OFFSET ?", (1, page * NUM_ROWS * NUM_COLUMNS + int(image_pos)))
    row = cursor.fetchone()
    if row:
        row_id = row[0]  # Assuming the first column is the primary key
        cursor.execute("UPDATE images SET rating = ? WHERE id = ?", (int(score), row_id))
        conn.commit()


def del_image(page, image_pos):
    page = page - 1
    conn = sqlite3.connect(os.path.join("flagged", "data.db"))
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM images LIMIT ? OFFSET ?", (1, page * NUM_ROWS * NUM_COLUMNS + int(image_pos)))
    row = cursor.fetchone()
    if row:
        row_id = row[0]  # Assuming the first column is the primary key
        cursor.execute("DELETE FROM images WHERE id = ?", (row_id,))
        os.remove(row[1])
        conn.commit()
        return update_gallery(page)


def update_gallery(page):
    to_return = []
    page = int(page) - 1
    conn = sqlite3.connect(os.path.join("flagged", "data.db"))
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM images LIMIT ? OFFSET ?", (NUM_ROWS * NUM_COLUMNS, page * NUM_ROWS * NUM_COLUMNS))
    rows = cursor.fetchall()
    for i in range(len(rows)):
        _, path, score = rows[i]
        to_return.append(gr.update(value=str(score)))
        to_return.append(Image.open(path))
    conn.close()
    while len(to_return) / 2 < NUM_ROWS * NUM_COLUMNS:
        to_return.append(None)
    return to_return


def prev_page(inp):
    if not inp <= 1:
        inp = inp - 1
    to_return = [inp]
    to_return.extend(update_gallery(inp))
    return to_return


def next_page(inp):
    to_return = [inp + 1]
    to_return.extend(update_gallery(inp + 1))
    return to_return


def build_gallery(name, rows, columns) -> dict:
    to_update = []
    to_return = {}
    del_buttons = []
    images_pos = []
    with gr.Tab(name) as tab:
        with gr.Row():
            prev = gr.Button("Previous page")
            page_counter = gr.Number(value=1, label="Current page", interactive=False)
            next = gr.Button("Next page")

        for i in range(rows):
            with gr.Row():
                for j in range(columns):
                    with gr.Column():
                        image_number = gr.Textbox(i * columns + j, visible=False)
                        delete_button = gr.Button("Delete")
                        to_return[f"pos_{i * columns + j}"] = gr.Image()
                        buttons = gr.Radio(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"], label="Rating",
                                           interactive=True)
                        buttons.associated_image = to_return[f"pos_{i * columns + j}"]
                        to_return[f"pos_{i * columns + j}"].associated_radio = buttons
                        rescore_button = gr.Button("Rescore")
                        rescore_button.click(rescore, inputs=[page_counter, image_number, buttons])

                        del_buttons.append(delete_button)
                        images_pos.append(image_number)
                        to_update.append(buttons)
                        to_update.append(to_return[f"pos_{i * columns + j}"])

        tab.select(update_gallery, inputs=page_counter, outputs=to_update)
        temp = [page_counter]
        temp.extend(to_update)
        next.click(next_page, inputs=page_counter, outputs=temp)
        prev.click(prev_page, inputs=page_counter, outputs=temp)
        for i,j in zip(del_buttons, images_pos):
            i.click(del_image, inputs=[page_counter, j],outputs=to_update)
    return to_return
