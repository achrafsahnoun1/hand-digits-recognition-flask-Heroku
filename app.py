from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import requests
import io



# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

MODEL_PATH = 'models/cnnfold7.pth'
# Load your trained model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64,64, kernel_size=5)
        self.conv3 = nn.Conv2d(64,128, kernel_size=5)
        self.fc1 = nn.Linear(3*3*128,256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1,3*3*128 )
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    
model = CNN()
model.load_state_dict(torch.load(MODEL_PATH,map_location='cpu'))
model.eval()

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(image, model):
    
    x=torch.from_numpy(np.array(image))
    x=x.view(-1,1,28,28).float()
  
    output = model(x)
    predicted_label=torch.max(output,1)[1]
    res=predicted_label.numpy()[0]
   
    return res

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        file = request.files.get('file')
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes))
 
        # Make prediction
        preds = model_predict(image, model)
        return str(preds)
    return None


if __name__ == '__main__':
    app.run(debug=True)

