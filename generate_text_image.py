import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw
# from skimage import io  # Not used
# from skimage.transform import resize # Not used
import textwrap

# Parameters
FONT_SIZE = 200
FONT_COLOR = (0, 0, 0)
INPUT_IMAGE_SIZE = (1000, 1000)  # Model's expected input size (height, width) - consistent with training
OUTPUT_IMAGE_SIZE = (1000, 1000)  # Enlarged output size for better quality
BACKGROUND_COLOR = (255, 255, 255)
MODEL_PATH = 'model/handwriting_model.h5'
OUTPUT_IMAGE = 'output/output_text.png'
FONT_FILE = "Myfont1-Regular.ttf"
MAX_WIDTH = 50  # max size of text


def create_text_image(text, font):
    img = Image.new('RGB', (INPUT_IMAGE_SIZE[1], INPUT_IMAGE_SIZE[0]), BACKGROUND_COLOR)  # width, height
    draw = ImageDraw.Draw(img)

    wrapped_text = textwrap.fill(text, width=30)  # Adjust width as needed
    lines = wrapped_text.split('\n')
    y_text = 50

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = ((INPUT_IMAGE_SIZE[1]) - text_width) / 2

        draw.text((x, y_text), line, font=font, fill=FONT_COLOR)
        y_text += text_height + 10
    return img


def generate_handwritten_text(text, model):
    font = ImageFont.truetype(FONT_FILE, size=FONT_SIZE)
    text_img = create_text_image(text, font)
    text_img_array = np.array(text_img).astype(np.float32) / 255.0
    text_img_array = np.expand_dims(text_img_array, axis=0)

    generated_img_array = model.predict(text_img_array)
    generated_img_array = np.squeeze(generated_img_array, axis=0)

    generated_img_array = np.clip(generated_img_array * 255, 0, 255).astype(np.uint8)

    generated_img = Image.fromarray(generated_img_array)
    generated_img = generated_img.resize(OUTPUT_IMAGE_SIZE, Image.LANCZOS)  # resized output

    return generated_img


def main():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    user_text = """Hello
               I am Risha Dey"""  #Added special characters
    generated_img = generate_handwritten_text(user_text, model)
    generated_img.save(OUTPUT_IMAGE)
    print(f"Generated image saved to {OUTPUT_IMAGE}")


if __name__ == '__main__':
    main()