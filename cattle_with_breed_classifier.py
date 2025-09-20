import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="🐄 Futuristic Cattle & Breed Classifier",
    layout="wide",
    page_icon="🐂"
)

# -----------------------------
# Custom CSS for styling
# -----------------------------
st.markdown("""
<style>
/* Global background gradient */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Card container */
.card {
    background: rgba(255, 255, 255, 0.05);
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 0 30px rgba(0,0,0,0.7);
    transition: transform 0.3s ease;
}
.card:hover {
    transform: scale(1.03);
}

/* Title styling */
h1, h2, h3, h4, h5, h6 {
    color: white;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #ff8c00, #ff0080);
    color: white;
    font-weight: bold;
    border-radius: 12px;
    height: 50px;
    width: 100%;
    font-size: 18px;
    transition: 0.3s;
}
.stButton>button:hover {
    opacity: 0.85;
    transform: scale(1.02);
}

/* Image container */
.img-container {
    border-radius: 25px;
    overflow: hidden;
    box-shadow: 0px 0px 35px rgba(0,0,0,0.8);
}

/* Progress bar container */
.progress-bar-container {
    background: rgba(255, 255, 255, 0.15);
    border-radius: 10px;
    width: 100%;
    height: 25px;
    margin-top: 10px;
}

/* Progress bar fill */
.progress-bar-fill {
    background: linear-gradient(90deg, #ff8c00, #ff0080);
    height: 100%;
    border-radius: 10px;
    text-align: center;
    line-height: 25px;
    color: white;
    font-weight: bold;
}

/* Description text */
.description {
    color: #e0e0e0;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Transform for images
# -----------------------------
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# -----------------------------
# Load Models
# -----------------------------
def load_cattle_model(model_path):
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def load_breed_model(model_path, num_classes):
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

# -----------------------------
# Class Names
# -----------------------------
cattle_class_names = ['Buffalo', 'Cow', 'None']
breed_names = ['Alambadi', 'Amritmahal', 'Ayrshire', 'Banni', 'Bargur', 'Bhadawari', 'Brown_Swiss', 'Dangi', 
               'Deoni', 'Gir', 'Guernsey', 'Hallikar', 'Hariana', 'Holstein_Friesian', 'Jaffrabadi', 'Jersey', 
               'Kangayam', 'Kankrej', 'Kasargod', 'Kenkatha', 'Kherigarh', 'Khillari', 'Krishna_Valley', 
               'Malnad_gidda', 'Mehsana', 'Murrah', 'Nagori', 'Nagpuri', 'Nili_Ravi', 'Nimari', 'Ongole', 
               'Pulikulam', 'Rathi', 'Red_Dane', 'Red_Sindhi', 'Sahiwal', 'Surti', 'Tharparkar', 'Toda', 
               'Umblachery', 'Vechur']

# -----------------------------
# Prediction Functions
# -----------------------------
def predict_cattle(image, model):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)
        return cattle_class_names[predicted.item()], confidence.item()

def predict_breed(image, model, breed_names):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return breed_names[predicted.item()]

# -----------------------------
# App Layout
# -----------------------------
st.title("🐄 Futuristic Cattle & Breed Classifier")
st.markdown('<p class="description">Upload an image of a cow or buffalo to classify it and detect its breed with high accuracy. The app uses advanced deep learning models for both cattle and breed detection.</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image", type=['jpg','jpeg','png'])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')

    # Display image in a card
    st.markdown('<div class="card img-container">', unsafe_allow_html=True)
    st.image(image, use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Load cattle model
    cattle_model_path = 'models/best_cow_buffalo_none_classifier.pth'
    cattle_model = load_cattle_model(cattle_model_path)

    # Predict cattle
    with st.spinner("Classifying cattle..."):
        predicted_cattle, confidence = predict_cattle(image, cattle_model)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Cattle Classification Result")
    st.write(f"**Predicted Cattle:** {predicted_cattle}")
    # Confidence bar
    st.markdown(f'''
    <div class="progress-bar-container">
        <div class="progress-bar-fill" style="width: {confidence*100:.2f}%">{confidence*100:.2f}%</div>
    </div>
    ''', unsafe_allow_html=True)

    if confidence >= 0.60 and predicted_cattle in ['Cow', 'Buffalo']:
        st.markdown('<p class="description">The confidence is high enough. You can now detect the breed of this cattle.</p>', unsafe_allow_html=True)

        if st.button("🚀 Detect Breed"):
            breed_model_path = 'models/breed_classifier.pth'
            breed_model = load_breed_model(breed_model_path, len(breed_names))

            with st.spinner(f"Detecting {predicted_cattle} breed..."):
                predicted_breed = predict_breed(image, breed_model, breed_names)

            st.balloons()
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Breed Detection Result")
            st.success(f"Detected Breed: {predicted_breed} 🏆")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning(f"Confidence too low ({confidence*100:.2f}%), classified as 'None'. Breed detection cannot proceed.")
    st.markdown('</div>', unsafe_allow_html=True)
