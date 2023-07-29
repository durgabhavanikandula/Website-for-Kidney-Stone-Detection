from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def perform_kidney_stone_detection(image_path):
    # Load the ultrasound image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform adaptive thresholding to create a binary image
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Select the largest contour (assuming it represents the required area)
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a binary mask with the same dimensions as the image
    mask = np.zeros_like(gray)

    # Draw the largest contour filled with white on the mask
    cv2.drawContours(mask, [largest_contour], -1, (255), cv2.FILLED)

    # Apply the mask to the original image
    segmented_image = cv2.bitwise_and(image, image, mask=mask)

    # Convert the segmented image to grayscale
    gray_segmented = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection using the Canny method
    edges = cv2.Canny(gray_segmented, 100, 200)

    # Apply thresholding using the binary method
    _, binary = cv2.threshold(gray_segmented, 127, 255, cv2.THRESH_BINARY)

    # Apply morphological operations to refine the binary image
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(binary, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=1)
    
    # Find contours in the dilated image
    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize a counter for the number of rectangles marked
    num_rectangles = 0

    # Iterate through each contour and mark the stones on the image
    marked_image = segmented_image.copy()  # Create a copy of the original image
    min_area_threshold = 100

    for contour in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)

        # Filter out small contours (noise) by setting a minimum area threshold
        if area < min_area_threshold:
            # Draw a bounding rectangle around the contour
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(marked_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Increment the counter for the number of rectangles marked
            num_rectangles += 1

    # Check the number of rectangles marked and store the result
    if num_rectangles > 0:
        num_rectangles_text = f"{num_rectangles} stones detected"
    else:
        num_rectangles_text = "No stones detected"
        return marked_image, num_rectangles_text, [], []
    
    # Find contours in the dilated image
    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through each contour and mark the stones on the image
    marked_image = segmented_image.copy()  # Create a copy of the original image
    min_area_threshold = 100.0
    stone_sizes = []  # List to store the sizes of detected stones
    stone_diameters = []  # List to store the diameters of detected stones

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > min_area_threshold:
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate the size of the stone based on the dimensions of the bounding rectangle
            stone_size = (w, h)
            stone_sizes.append(stone_size)

            # Calculate the diameter of the stone based on the dimensions of the bounding rectangle
            stone_diameter = max(w, h)
            stone_diameters.append(stone_diameter)

    # Store the sizes of the detected stones with width and height greater than 50
    stone_sizes_text = ""
    for i, size in enumerate(stone_sizes):
        if size[0] > 50 and size[1] > 50:
            x, y, w, h = cv2.boundingRect(contours[i])
            cv2.rectangle(marked_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
            stone_sizes_text += "Stone: Width = {}, Height = {}\n".format(size[0], size[1])

    # Store the diameters of the detected stones with width and height greater than 50
    stone_diameters_text = ""
    for i, diameter in enumerate(stone_diameters):
        if stone_sizes[i][0] > 50 and stone_sizes[i][1] > 50:
            stone_diameters_text += "Stone: Diameter = {}mm\n".format(diameter)

    return marked_image, num_rectangles_text, stone_sizes, stone_diameters


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/stone-detection', methods=['POST'])
def stone_detection():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        marked_image, num_rectangles, stone_sizes, stone_diameters = perform_kidney_stone_detection(filepath)

        output_filename = 'output.jpg'
        output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        cv2.imwrite(output_filepath, marked_image)

        stones = []
        for i in range(len(stone_sizes)):
            if stone_sizes[i][0] > 50 and stone_sizes[i][1] > 50:
                stone = {
                    'size': {
                        'width': stone_sizes[i][0],
                        'height': stone_sizes[i][1]
                    },
                    'diameter': stone_diameters[i]
                }
                stones.append(stone)

        return jsonify({
            'output_image_url': f'/uploads/{output_filename}',
            'num_rectangles': num_rectangles,
            'stones': stones
        })

    return jsonify({'error': 'Invalid file format'})



if __name__ == '__main__':
    app.run()
