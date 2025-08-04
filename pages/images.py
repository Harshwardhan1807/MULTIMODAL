import easyocr
from PIL import Image
import numpy as np
import re
reader = easyocr.Reader(['en'])

import streamlit as st

st.set_page_config(
    page_title="Images",
    page_icon="🖼️"
)

st.title("Images")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    results = reader.readtext(image_np)
    text = "\n".join([res[1] for res in results])
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  
    text = re.sub(r'\n{3,}', '\n\n', text)
    st.write(text)