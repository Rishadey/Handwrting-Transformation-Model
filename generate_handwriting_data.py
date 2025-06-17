import os
from PIL import Image, ImageFont, ImageDraw
import numpy as np

# Parameters
FONT_SIZE = 50
FONT_COLOR = (0, 0, 0)
IMAGE_SIZE = (1000, 1000)  # Height, Width
LINE_SPACING = 10
BACKGROUND_COLOR = (255, 255, 255)
OUTPUT_FOLDER = 'output/train_images'
FONT_FILE = "Myfont1-Regular.ttf"  # Corrected font name

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def create_text_image(text, font):
    img = Image.new('RGB', IMAGE_SIZE, BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    try:  # Add error handling
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except Exception as e:
        print(f"Error getting textbbox for '{text}': {e}")  # Debug
        return None  # Or handle the error differently

    x = (IMAGE_SIZE[1] - text_width) / 2  # width - text
    y = (IMAGE_SIZE[0] - text_height) / 2  # height - text
    draw.text((x, y), text, font=font, fill=FONT_COLOR)
    return img


def is_valid_filename(filename):
    """Checks if a filename is valid for most operating systems."""
    # Replace invalid characters with an underscore or skip the file
    invalid_characters = r'<>:"/\|?*'  #Add '#' and '@' if it doesn't work
    for char in invalid_characters:
        if char in filename:
            return False
    return True

def main():
    font = ImageFont.truetype(FONT_FILE, size=FONT_SIZE)

    with open("labels.txt", "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    print(f"Number of lines read from labels.txt: {len(lines)}") #DEBUG

    for i, text in enumerate(lines):
        text = text.strip()  # Ensure no leading/trailing whitespace #STRIP AGAIN
        if not text:
            print(f"Skipping empty line at index {i}") #DEBUG
            continue

        if not is_valid_filename(text):
            print(f"Skipping invalid filename: '{text}' at index {i}") #DEBUG
            continue

        img = create_text_image(text, font)

        if img is None:
            print(f"Skipping saving image for '{text}' due to error in create_text_image") #DEBUG
            continue

        filename = os.path.join(OUTPUT_FOLDER, f'{text}.png')
        try:
            img.save(filename)  # Save with label as filename
            print(f"Generated image for: '{text}' saved as '{filename}'") #DEBUG
        except Exception as e:
            print(f"Error saving image for '{text}': {e}") #More verbose error message



if __name__ == '__main__':
    main()