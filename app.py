from flask import Flask, render_template, redirect, request, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename
import torch
import cv2
import supervision as sv
from segment_anything import SamPredictor, sam_model_registry
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import os
import logging
import pdb
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_DIRECTORY'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB
app.config['ALLOWED_EXTENSIONS'] = ['.jpg', '.jpeg', '.png', '.gif', '.JPG', '.JPEG', '.PNG', '.GIF']

@app.route('/')
def index():
  app.logger.setLevel(logging.DEBUG)
  app.logger.debug('Script Started!')
  files = os.listdir(app.config['UPLOAD_DIRECTORY'])
  images = []
  

  for file in files:
    #if os.path.splitext(file)[1].lower() in app.config['ALLOWED_EXTENSIONS']:
    if os.path.splitext(file)[1] in app.config['ALLOWED_EXTENSIONS']:
      images.append(file)
  
  return render_template('index.html', images=images)

@app.route('/upload', methods=['POST'])
def upload():
  try:
    file = request.files['file']

    if file:
      # extension = os.path.splitext(file.filename)[1].lower()
      extension = os.path.splitext(file.filename)[1]

      if extension not in app.config['ALLOWED_EXTENSIONS']:
        return 'File is not an image.'
      
      file.save(os.path.join(app.config['UPLOAD_DIRECTORY'], secure_filename(file.filename)))
      
      
      MODEL_TYPE = 'vit_h'
      sam = sam_model_registry[MODEL_TYPE](checkpoint='sam_vit_h_4b8939.pth')
      mask_generator = SamAutomaticMaskGenerator(sam)
      base_dir = os.path.dirname(os.path.abspath(__file__))
      target_file_dir = os.path.join(app.config['UPLOAD_DIRECTORY'], secure_filename(file.filename))
      app.logger.debug(base_dir)
      image = cv2.imread(os.path.join(base_dir, target_file_dir))
      # app.logger.debug(type(image))
      app.logger.debug('Generating Mask!')
      masks = mask_generator.generate(image)
      
      # Visualise SAM masks
      mask_annotator = sv.MaskAnnotator()
      detections = sv.Detections.from_sam(sam_result=masks)
      annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)  
      cv2.imwrite(os.path.join(app.config['UPLOAD_DIRECTORY'], secure_filename(file.filename)), annotated_image)
      
  
  except RequestEntityTooLarge:
    return 'File is larger than the 16MB limit.'
  
  return redirect('/')

@app.route('/serve-image/<filename>', methods=['GET'])
def serve_image(filename):
    return send_from_directory(app.config['UPLOAD_DIRECTORY'], filename)

if __name__ == '__main__':
    app.debug = True
    app.run()

