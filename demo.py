import requests
from keys import openai_key
import streamlit as st
from openai import OpenAI
from mesh_maker import MeshMaker

client = OpenAI(api_key=openai_key)
state = st.session_state

st.title("Model Generator")

if "image" not in state:
    state.image = None
if "maker" not in state:
    with st.spinner("Loading Models..."):
        state.maker = MeshMaker()

with st.expander("Meta Prompt"):
    meta_prompt = st.text_area("The meta prompt to wrap user's prompt", value="<user_prompt>; all objects together; smooth 3d rendering in metal; colored black; uniform flat white background; frontal view; head on")
    meta_prompt = meta_prompt.strip()

def get_image(prompt):
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    image_url =response.data[0].url
    image = requests.get(image_url).content
    return image


user_prompt = st.text_area("User Prompt", value="A chess piece with the head of a wolf")
user_prompt = user_prompt.strip()

img_btw = st.button("Generate Image")
if img_btw:
    prompt = meta_prompt.replace("<user_prompt>", user_prompt)
    st.write(f"Prompt: {prompt}")
    with st.spinner("Generating Image..."):
        image = get_image(prompt)
        state.image = image
        # write to file
        with open("examples/image.png", "wb") as f:
            f.write(image)
        st.experimental_rerun()

if state.image is not None:
    st.image(state.image, caption="Generated Image", use_column_width=True)
    mesh_btn = st.button("Generate Mesh")
    if mesh_btn:
        progress = st.progress(0, text="Generating Mesh")
        path = state.maker.make("examples/image.png", progress)
        st.download_button("Download Mesh", path)







