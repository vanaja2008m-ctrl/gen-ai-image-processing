import streamlit as st
import torch
from diffusers import DiffusionPipeline

st.title("🎨 AI Image Generator")

prompt = st.text_input(
    "Enter your prompt:",
    "A cute cat sitting in a garden"
)

if st.button("Generate Image"):

    with st.spinner("Generating image..."):

        pipe = DiffusionPipeline.from_pretrained(
            "segmind/tiny-sd",
             torch_dtype=torch.float32
        )

        image = pipe(prompt).images[0]

        st.image(
            image,
            caption="Generated Image"
        )