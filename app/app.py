from flask import Flask, render_template, request, session, redirect, url_for
from flask_session import Session
import os
from werkzeug.utils import secure_filename
import sys
import matplotlib.pyplot as plt
from torch import zeros

sys.path.append(r'C:\Users\User\Desktop\Ahmad\Hackathon\app')


# Defining upload folder path
UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
# # Define allowed files
# ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
 
# The default folder name for static files should be "static" else need to mention custom folder for static path
app = Flask(__name__, template_folder='Templates', static_folder='staticFiles')

# Configure upload folder for Flask application
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"

Session(app)
 

@app.route('/',  methods=("POST", "GET"))
def uploadFile():
    if request.method == 'POST':
        # take the two models as global variables in order to avoid importing them each time
        global classifier
        global mask

        # Upload file flask
        uploaded_img = request.files['uploaded-file']

        # Extracting uploaded data file name
        img_filename = secure_filename(uploaded_img.filename)

        # Storing uploaded file path in flask session
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)

        # Upload file to database (defined uploaded folder in static path)
        uploaded_img.save(img_path)

        # Find label for the image
        label = classifier.labelise(img_path)

        # Find mask if the image contains silo
        if label == 1:
            pred_mask = mask.generate_mask(img_path)
        else:
            pred_mask = zeros((256,256))
        mask_name = "_mask_"+img_filename

        # Save the mask
        fig = plt.figure(frameon=False)
        plt.axis('off')
        plt.imshow(pred_mask, cmap = 'gray')
        mask_path = os.path.join(app.config['UPLOAD_FOLDER'], "mask", mask_name)
        plt.savefig(mask_path)
        return redirect(url_for('labeled', label=label,img_path=img_path, mask_path=mask_path))
    else:
        return render_template('index_upload.html', color= "blue")


@app.route('/labeled')
def labeled():

    # Afficher les resultats
    label = request.args['label']
    img_path = request.args['img_path']
    mask_path = request.args['mask_path']
    return render_template('show_image.html', img_path = img_path, label=label, mask_path=mask_path)

 
if __name__=='__main__':
    from script.model import Model 
    classifier = Model()
    
    from script.segment import Mask
    mask = Mask()

    port = int(os.environ.get("PORT", 5000))
    app.run(debug = True, port=port)