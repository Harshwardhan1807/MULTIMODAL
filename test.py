import easyocr
from PIL import Image
import numpy as np
import re
reader = easyocr.Reader(['en'])  # Add more languages if needed
image = Image.open(r"C:\Users\shars\Downloads\Scanned Report_page-0001.jpg")
image_np = np.array(image)

results = reader.readtext(image_np)
text = "\n".join([res[1] for res in results])
text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  
text = re.sub(r'\n{3,}', '\n\n', text)
print(text)
