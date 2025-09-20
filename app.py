from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import base64
from io import BytesIO

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Production configuration
if os.environ.get('FLASK_ENV') == 'production':
    app.config['DEBUG'] = False
    app.config['TESTING'] = False
    app.config['SERVER_NAME'] = os.environ.get('RENDER_EXTERNAL_URL', '').replace('https://', '').replace('http://', '')
else:
    app.config['DEBUG'] = True

# Ensure the upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -----------------------------
# Model Configuration
# -----------------------------
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Class names
cattle_class_names = ['Buffalo', 'Cow', 'None']
breed_names = ['Alambadi', 'Amritmahal', 'Ayrshire', 'Banni', 'Bargur', 'Bhadawari', 'Brown_Swiss', 'Dangi', 
               'Deoni', 'Gir', 'Guernsey', 'Hallikar', 'Hariana', 'Holstein_Friesian', 'Jaffrabadi', 'Jersey', 
               'Kangayam', 'Kankrej', 'Kasargod', 'Kenkatha', 'Kherigarh', 'Khillari', 'Krishna_Valley', 
               'Malnad_gidda', 'Mehsana', 'Murrah', 'Nagori', 'Nagpuri', 'Nili_Ravi', 'Nimari', 'Ongole', 
               'Pulikulam', 'Rathi', 'Red_Dane', 'Red_Sindhi', 'Sahiwal', 'Surti', 'Tharparkar', 'Toda', 
               'Umblachery', 'Vechur']

# -----------------------------
# Model Loading Functions
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

# Load models globally
cattle_model = load_cattle_model('models/best_cow_buffalo_none_classifier.pth')
breed_model = load_breed_model('models/breed_classifier.pth', len(breed_names))

# -----------------------------
# Prediction Functions
# -----------------------------
def predict_cattle(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = cattle_model(image)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)
        return cattle_class_names[predicted.item()], confidence.item()

def predict_breed(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = breed_model(image)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)
        return breed_names[predicted.item()], confidence.item()

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def home():
    """Home page route"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({'error': 'Invalid file format'}), 400

        # Read and process the image
        image_data = file.read()
        image = Image.open(BytesIO(image_data)).convert('RGB')
        
        # Make predictions
        cattle_type, cattle_confidence = predict_cattle(image)
        
        # Only predict breed if it's a cow or buffalo with high confidence
        breed_result = None
        if cattle_confidence >= 0.60 and cattle_type in ['Cow', 'Buffalo']:
            breed_name, breed_confidence = predict_breed(image)
            breed_result = {
                'name': breed_name,
                'confidence': f"{breed_confidence*100:.2f}%"
            }
        
        # Convert image to base64 for display
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'cattle_type': cattle_type,
            'cattle_confidence': f"{cattle_confidence*100:.2f}%",
            'breed_result': breed_result,
            'image': img_str
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/about')
def about():
    """About page route"""
    return render_template('about.html')

@app.route('/faq')
def faq():
    """FAQ page route"""
    return render_template('faq.html')

@app.route('/guide')
def guide():
    """User guide page route"""
    return render_template('guide.html')

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    return render_template('errors/404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template('errors/500.html'), 500

if __name__ == '__main__':
    # Get port from environment variable or use 5000 as default
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)