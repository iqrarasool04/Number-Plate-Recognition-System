import os
import cv2
import numpy as np
import pandas as pd
import imutils
import matplotlib.pyplot as plt
import pytesseract as pt
import easyocr
import plotly.express as px
# pt.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\tesseract.exe'  
import skimage.io
from imutils.video import VideoStream
import keyboard
import time

INPUT_WIDTH =  640
INPUT_HEIGHT = 640


#loading trained yolo model
model = cv2.dnn.readNetFromONNX('C:\\Users\\DELL\\Documents\\GitHub\\Number-Plate-Recognition-System\\weights\\best.onnx') 
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)



def get_detections(img,model):

    #converting images to yolo format
    image = img.copy()
    row, col, d = image.shape

    max_rc = max(row,col)
    input_image = np.zeros((max_rc,max_rc,3),dtype=np.uint8)
    input_image[0:row,0:col] = image

    #getting predictions from yolo model
    blob = cv2.dnn.blobFromImage(input_image,1/255,(INPUT_WIDTH,INPUT_HEIGHT),swapRB=True,crop=False)
    model.setInput(blob)
    preds = model.forward()
    detections = preds[0]
    
    return input_image, detections
    


#for post processing of results
def non_maximum_supression(input_image,detections):
    
    #detecting filters based on models and probability
    boxes = []
    confidences = []

    image_w, image_h = input_image.shape[:2]
    x_factor = image_w/INPUT_WIDTH
    y_factor = image_h/INPUT_HEIGHT

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4] #confidence of detecting license plate
        if confidence > 0.4:
            class_score = row[5] #probability score of license plate
            if class_score > 0.25:
                cx, cy , w, h = row[0:4]
                left = int((cx - 0.5*w)*x_factor)
                top = int((cy-0.5*h)*y_factor)
                width = int(w*x_factor)
                height = int(h*y_factor)
                box = np.array([left,top,width,height])
                confidences.append(confidence)
                boxes.append(box)

    #cleaning
    boxes_np = np.array(boxes).tolist() 
    confidences_np = np.array(confidences).tolist()
    
    #NMS
    index = cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.25,0.45)
    
    return boxes_np, confidences_np, index



def drawings(image,boxes_np,confidences_np,index):
    # Drawings
    for ind in index:
        x,y,w,h =  boxes_np[ind]
        bb_conf = confidences_np[ind]
        conf_text = 'plate: {:.0f}%'.format(bb_conf*100)
        
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.rectangle(image,(x,y-30),(x+w,y),(255,0,255),-1)
        cv2.putText(image,conf_text,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),1)



# predictions flow with return result
def yolo_predictions(img,model):
    # detections
    input_image, detections = get_detections(img,model)
    # NMS
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)
    # Drawings
    result_img = drawings(img,boxes_np,confidences_np,index)

    global box_coordinate
    global nm_index
    box_coordinate = boxes_np
    nm_index = index

    
    return result_img



   
def croptheROI(image,bbox, index):
     
    cropped = None
    for i in index:   
        x,y,w,h =  bbox[i]
        cropped = image[y:y+h, x:x+w]
        # Check if 'cropped' is not None 
        if cropped is not None:
          cv2.imwrite('cropped.png', cropped)

    return cropped


# def preprocessing(crop):
#     gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

#     bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
#     edged = cv2.Canny(bfilter, 30, 200)  # Edge detection

#     keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     contours = imutils.grab_contours(keypoints)
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

#     if not contours:
#         # handle case of no contours found 
#         print("No contours found.")
#         return None

#     location = None
#     for contour in contours:
#         approx = cv2.approxPolyDP(contour, 10, True)
#         if len(approx) == 4:
#             location = approx
#             break

#     if location is not None:
#         mask = np.zeros(gray.shape, np.uint8)
#         new_image = cv2.drawContours(mask, [location], 0, 255, -1)
#         new_image = cv2.bitwise_and(crop, crop, mask=mask)

#         (x, y) = np.where(mask == 255)
#         (x1, y1) = (np.min(x), np.min(y))
#         (x2, y2) = (np.max(x), np.max(y))
#         cropped_image = gray[x1:x2 + 1, y1:y2 + 1]

#         return cropped_image
#     else:
#         # handle case of invalid contours found 
#         print("Invalid contour found.")
#         return None




# def extract_text(cropped_image):
    
#     reader = easyocr.Reader(['en'])
#     result = reader.readtext(cropped_image)
    
    
#     return result
    
# def extract_text(cropped_image):
#     # Check if cropped_image is not None
#     if cropped_image is None:
#         print("No valid cropped image.")
#         return None

#     # Convert NumPy array to bytes
#     _, buffer = cv2.imencode('.jpg', cropped_image)
#     image_bytes = buffer.tobytes()

#     # Use easyocr to extract text
#     reader = easyocr.Reader(['en'])
#     result = reader.readtext(image_bytes)

#     return result



# Read the original image
original_image = cv2.imread('C:\\Users\\DELL\\Documents\\GitHub\\Number-Plate-Recognition-System\\samples\\seecsCar3.jpeg')  

# Resize the image
resized_image = cv2.resize(original_image, (1200, 1200))

# Display the original and resized images using Matplotlib
# plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
# plt.subplot(1, 2, 2), plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)), plt.title('Resized Image')
# plt.show()

# test
results = yolo_predictions(resized_image, model)
roi = croptheROI(resized_image, box_coordinate, nm_index)
# Check if 'roi' is not None before proceeding with further processing
if roi is not None:
    # pp_image = preprocessing(roi)
    # text = extract_text(pp_image)
    # print(text)
    fig = px.imshow(resized_image)
    fig.update_layout(width=700, height=400, margin=dict(l=10, r=10, b=10, t=10))
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    fig.show()
else:
    print("No contours found in the cropped image.")


import plotly.io as ip
ip.renderers.default='browser'

