import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
import torch.nn.functional as function
from torchvision import transforms
from torchvision.models import densenet161
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Loaded trained model
model_path = 'model_custom.pt'
num_classes = 4  
class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']  

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class alzhemiers_Model(nn.Module):
    def __init__(self):
        super(alzhemiers_Model,self).__init__()
        self.convolution_layer1 = nn.Conv2d(in_channels=3,out_channels=7,kernel_size = 3,stride=1)
        self.bn1 = nn.BatchNorm2d(7)
        self.convolution_layer2 = nn.Conv2d(in_channels=7,out_channels=17,kernel_size=3,stride=1)
        self.bn2 = nn.BatchNorm2d(17)
        self.convolution_layer3 = nn.Conv2d(in_channels=17,out_channels=37,kernel_size = 3,stride=1)
        self.bn3 = nn.BatchNorm2d(37)
        self.convolution_layer4 = nn.Conv2d(in_channels=37,out_channels=73,kernel_size=3,stride=1)
        self.bn4 = nn.BatchNorm2d(73)
        self.fc1 = nn.Linear(in_features=14*14*73,out_features=8000)
        self.fc2 = nn.Linear(in_features = 8000,out_features = 3000)
        self.fc3 = nn.Linear(in_features = 3000,out_features = 500)
        self.fc4 = nn.Linear(in_features = 500,out_features = 30)
        self.fc5 = nn.Linear(in_features = 30 ,out_features = 4)

    def forward(self,x):
        x = self.bn1(function.relu(self.convolution_layer1(x)))
        x = torch.max_pool2d(x,kernel_size=2,stride =2)
        x = self.bn2(function.relu(self.convolution_layer2(x)))
        x = torch.max_pool2d(x,kernel_size=2,stride =2)
        x = self.bn3(function.relu(self.convolution_layer3(x)))
        x = torch.max_pool2d(x,kernel_size=2,stride =2)
        x = self.bn4(function.relu(self.convolution_layer4(x)))
        x = torch.max_pool2d(x,kernel_size=2,stride =2)

        #to flatten rather than flatten() we use x.view(-1,size)
        x = x.view(-1,14308)

        #passing to fully connected layers
        x = function.relu(self.fc1(x))
        x = function.relu(self.fc2(x))
        x = function.relu(self.fc3(x))
        x = function.relu(self.fc4(x))
        x = self.fc5(x)

        return function.log_softmax(x,dim=1)
    
model = alzhemiers_Model()
model_state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(model_state_dict)
model.to(device)
model.eval()

# Define the preprocessing transformation
preprocess = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image.to(device)  # Move to the appropriate device (CPU or GPU)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/runtime')
def runtime():
    return render_template('runtime.html')

@app.route('/more')
def more():
    return render_template('more.html')

@app.route('/aboutus')
def about_us():
    return render_template('aboutus.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        file_path = os.path.join('uploads', filename)
        file.save(file_path)
        # Preprocess the image
        image = preprocess_image(file_path)

        # Predict
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            predicted_class = class_names[predicted.item()]

        return render_template('runtime.html', prediction=predicted_class)

    return jsonify({'error': 'Invalid file type'})

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
