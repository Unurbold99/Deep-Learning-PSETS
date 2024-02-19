import streamlit as st
from fastai.vision.all import *
import gdown

st.markdown("""# Mongolian Food Classifier

Mongolia has its unique dishes that vary in their looks and taste, this classifier can identify 4 different dishes which are "Khuushur, Buuz, Niislel salad, and Tsuivan". Please upload a picture to try it out.""")

st.markdown("""### Upload your image here""")

image_file = st.file_uploader("Image Uploader", type=["png","jpg","jpeg"])

## Model Loading Section
model_path = Path("export.pkl")

if not model_path.exists():
    with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
        url = 'https://drive.google.com/drive/u/2/folders/1YNqHfZ_chx0jTXDT1Gd-g6NOc0fv_DXk'
        output = 'export.pkl'
        gdown.download(url, output, quiet=False)
    learn_inf = load_learner('export.pkl')

col1, col2 = st.columns(2)
if image_file is not None:
    img = PILImage.create(image_file)
    pred, pred_idx, probs = learn_inf.predict(img)

    with col1:
        st.markdown(f"""### Predicted animal: {pred.capitalize()}""")
        st.markdown(f"""### Probability: {round(max(probs.tolist()), 3) * 100}%""")
    with col2:
        st.image(img, width=300)