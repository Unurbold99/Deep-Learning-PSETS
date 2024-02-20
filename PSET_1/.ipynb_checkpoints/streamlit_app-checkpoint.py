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
        url = 'https://drive.google.com/uc?id=1BWOq5H8rCSwn8hs96oo_Rs-9v3f3ftje'
        output = 'export.pkl'
        gdown.download(url, output, quiet=False)
    learn_inf = load_learner('export.pkl')
else:
    learn_inf = load_learner('export.pkl')

if image_file is not None:
    img = PILImage.create(image_file)
    pred, pred_idx, probs = learn_inf.predict(img)

    st.image(img, width=300, caption='Uploaded Image', use_column_width='always', output_format='auto')

    probability = round(max(probs.tolist()) * 100, 2)
    probability_color = 'red' if probability < 50 else ('yellow' if probability < 80 else 'green')

    st.markdown(f"""### Predicted food: {pred.capitalize()}""")
    if pred == 'бууз':
        st.write("Cornerstone of Mongolia's cuisine, a form of steamed dumplings that primarily is made of lamb and is often the main dish during Mongolia's holidays")
    elif pred == 'хуушуур':
        st.write("A deep fried slab of dough that has ground meat usually lamb inside it")
    elif pred == 'цуйван':
        st.write("A dish that is stir fried which includes doughy noodles with veggies and meat")
    elif pred == 'нийслэл салат':
        st.write("A salad that is rich in its flavor and calories, it includes potatoes, eggs, pickles, ham, carrots, corn and peas")

    st.markdown(f"""### Probability: <span style='color: {probability_color};'>{probability}%</span>""", unsafe_allow_html=True)
