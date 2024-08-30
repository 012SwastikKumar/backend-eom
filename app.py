from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Import CORS

import clip
from PIL import Image
import os
import torch
import json
import urllib.parse

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins
# CORS(app)  # Apply CORS to the entire app

# Load the device
#device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the custom-trained CLIP model
model, preprocess = clip.load("ViT-B/32", device=device)
model.load_state_dict(torch.load('clip_model.pth', map_location=device), strict=False)
model.to(device)
model.eval()

# Rename Files
def rename_files(directory):
    # List all files in the directory
    files = os.listdir(directory)
    
    # Sort files to ensure they are renamed in order
    files.sort()
    
    # Loop through all the files
    for index, filename in enumerate(files):
        # Get the file extension
        file_extension = os.path.splitext(filename)[1]
        
        # Construct the new file name with the prefix "img" followed by an increasing number
        new_filename = f"img{index + 1}{file_extension}"
        
        # Get the full path for the old and new file names
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_filename)
        
        # Rename the file
        os.rename(old_file, new_file)
        
    # print(f"Renamed {len(files)} files.")
# directory_path = "./images/"
# rename_files(directory_path)



# Path to your images directory
image_dir = './images'  # Assuming your images are in the "images" folder

# @app.get('/')
# def home():
    # print(device)
    # return render_template("index.html", image=0)

@app.route('/', methods=['GET'])
def get_data():
    data = {'message': 'Hello from Flask!'}
    return jsonify(data)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the prompt from the request body
    data = request.json
    prompt = data.get('prompt')
    
    print(prompt)
    print(device)

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    # Encode the text prompt
    text_features = model.encode_text(clip.tokenize(prompt).to(device))
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # # Initialize variables to store the best match
    # best_image = None
    # best_similarity = -1

    # Initialize a list to store image names and their similarity scores
    image_scores = []


    # # Iterate through images in the directory
    # for image_name in os.listdir(image_dir):
    #     image_path = os.path.join(image_dir, image_name)

    #     # Load and preprocess the image
    #     image = Image.open(image_path)
    #     image_input = preprocess(image).unsqueeze(0).to(device)

    #     # Encode the image
    #     image_features = model.encode_image(image_input)
    #     image_features /= image_features.norm(dim=-1, keepdim=True)

    #     # Calculate similarity
    #     similarity = torch.matmul(text_features, image_features.T).item()

    #     # Check if this is the best match
    #     if similarity > best_similarity:
    #         best_similarity = similarity
    #         best_image = image_name

     # Iterate through images in the directory
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)

        # Load and preprocess the image
        image = Image.open(image_path)
        image_input = preprocess(image).unsqueeze(0).to(device)

        # Encode the image
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Calculate similarity
        similarity = torch.matmul(text_features, image_features.T).item()

        # Append the image name and its similarity score to the list
        image_scores.append((image_name, similarity))
        # Sort the list by similarity scores in descending order
    image_scores.sort(key=lambda x: x[1], reverse=True)

    # Select the top 4 images
    top_images = image_scores[:4]

    # Create the response with the top 4 images and their similarity scores
    if top_images:
        torch.cuda.empty_cache()
        result = [{"image_url": f"./images/{urllib.parse.quote(img)}", "similarity_score": score} for img, score in top_images]
        return jsonify({"top_images": result}), 200
    # # Return the best matching image and its similarity score
    # if best_image:
    #     torch.cuda.empty_cache()
    #     # image_url = f"../../../../../images/{urllib.parse.quote(best_image)}"
    #     image_url = f"./images/{urllib.parse.quote(best_image)}"
    #     return jsonify({"best_image": image_url, "similarity_score": best_similarity}), 200
    else:
        torch.cuda.empty_cache()
        return jsonify({"error": "No images found"}), 404







# @app.route('/predict/<prompt>', methods=['GET'])
# def predict(prompt):
    # Get the prompt from the request
    print(prompt)
    print(device)
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    # Encode the text prompt
    text_features = model.encode_text(clip.tokenize(prompt).to(device))
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Initialize variables to store the best match
    best_image = None
    best_similarity = -1

    # Iterate through images in the directory
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)

        # Load and preprocess the image
        image = Image.open(image_path)
        image_input = preprocess(image).unsqueeze(0).to(device)

        # Encode the image
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Calculate similarity
        similarity = torch.matmul(text_features, image_features.T).item()

        # Check if this is the best match
        if similarity > best_similarity:
            best_similarity = similarity
            best_image = image_name

    # Return the best matching image and its similarity score
    if best_image:
        torch.cuda.empty_cache()
        image_url = f"/images/{urllib.parse.quote(best_image)}"
        return jsonify({"best_image": image_url, "similarity_score": best_similarity}), 200
        # print(os.path.join(image_dir, best_image))
        # return render_template("index.html", image= "images/"+urllib.parse.quote(best_image))
    else:
        torch.cuda.empty_cache()
        return jsonify({"error": "No images found"}), 404



if __name__ == '__main__':
    app.run(debug=True)


