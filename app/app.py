import streamlit as st
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Initialize MTCNN and InceptionResnetV1
mtcnn = MTCNN(keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def euclidean_distance(a, b):
    return (a - b).norm().item()

def cosine_distance(a, b):
    cosim = F.cosine_similarity(a.view(-1), b.view(-1), dim=0).item()
    return (cosim + 1) / 2 * 100

def compare_faces(img1, img2, edist_treshold, csim_treshold):
    """
    :param img1: PIL Image
    :param img2: PIL Image
    :param edist_treshold: Euclidean distance threshold
    :param csim_treshold: Cosine similarity threshold
    :return: Euclidean distance between img1 and img2
    """

    edict = dict(e1=None, e2=None)
    annotated_images = []

    for img in [img1, img2]:
        image = img.copy()
        faces, _ = mtcnn.detect(image)

        if faces is not None:
            for face in faces:
                draw = ImageDraw.Draw(image)
                draw.rectangle(face.tolist(), outline='yellow', width=3)
            annotated_images.append(image)

        aligned_image = mtcnn(image)
        if aligned_image is not None:
            embed = resnet(aligned_image)
            if edict['e1'] is None:
                edict['e1'] = embed
            else:
                edict['e2'] = embed

    edist = euclidean_distance(edict['e1'], edict['e2'])
    if edist < edist_treshold:
        st.success(f"Passed euclidean distance test with edist={edist}")
    else:
        st.error(f"Failed euclidean distance test with edist={edist}")

    csim_percent = cosine_distance(edict['e1'], edict['e2'])
    if csim_percent > csim_treshold:
        st.success(f"Passed cosine similarity percentage test with csim={csim_percent}")
    else:
        st.error(f"Failed cosine similarity percentage test with csim={csim_percent}")

    return edist, annotated_images, edict['e1'], edict['e2']

def plot_cosine_similarity(embed1, embed2):
    cos_sim = F.cosine_similarity(embed1.unsqueeze(0), embed2.unsqueeze(0), dim=1).detach().numpy()
    cos_sim_matrix = np.outer(cos_sim, cos_sim)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cos_sim_matrix, annot=True, cmap='coolwarm', cbar=True, square=True)
    plt.title('Pairwise Cosine Similarity')
    st.pyplot(plt)

# Streamlit app
st.title("Face Comparison App")

st.markdown("""
### Introduction
This app uses face detection and recognition models to compare two images. Upload clear images of faces for best results.

### What is an Embedding?
An embedding is a representation of an image (or other data) in a high-dimensional space. In this context, it captures the essential features of a face, allowing us to compare different faces.

### Euclidean Distance
The Euclidean distance between two points in a high-dimensional space is a measure of the straight-line distance between them.
- **Formula**: 
  $$d(x, y) = \\sqrt{\\sum_{i=1}^{n} (x_i - y_i)^2}$$
- **Interpretation**: Smaller distances indicate higher similarity.

### Cosine Similarity
Cosine similarity measures the cosine of the angle between two vectors in a high-dimensional space.
- **Formula**: 
  $$\\text{cosine\_similarity}(A, B) = \\frac{A \\cdot B}{||A|| \\cdot ||B||}$$
- **Interpretation**: Higher values (closer to 1 or 100%) indicate higher similarity.

### Set Thresholds
Use the sliders below to set the thresholds for Euclidean distance and cosine similarity. Adjust them based on the similarity you expect between the images.
""")

# Sliders for threshold settings
edist_treshold = st.slider("Set Euclidean Distance Threshold", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
csim_treshold = st.slider("Set Cosine Similarity Threshold (%)", min_value=0, max_value=100, value=80, step=1)

# File uploaders
img1 = st.file_uploader("Upload first image", type=["jpg", "jpeg", "png"])
img2 = st.file_uploader("Upload second image", type=["jpg", "jpeg", "png"])

if img1 and img2:
    # Convert uploaded files to PIL Images
    image1 = Image.open(img1)
    image2 = Image.open(img2)

    # Compare faces
    distance, annotated_images, embed1, embed2 = compare_faces(image1, image2, edist_treshold, csim_treshold)

    # Display images side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(annotated_images[0], caption="Image 1", use_column_width=True)
    with col2:
        st.image(annotated_images[1], caption="Image 2", use_column_width=True)

    # Plot pairwise cosine similarity
    plot_cosine_similarity(embed1, embed2)

st.info("Note: This app uses face detection and recognition models. Upload clear images of faces for best results.")
