import tensorflow_hub as hub
import cv2
import numpy
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

target_width = 1028
target_height = 1028
read_img = cv2.imread('amr_images/object_scenario_2.jpg')
inp = cv2.resize(read_img, (target_width , target_height ))
rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
image_rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)
image_rgb_tensor = tf.expand_dims(image_rgb_tensor , 0)
tfLite2Model = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")
labels = pd.read_csv('labels.csv', sep=';', index_col='ID')
labels = labels['OBJECT (2017 REL.)']

boxes, scores, classes, num_detections = tfLite2Model(image_rgb_tensor)

predictionLabels = classes.numpy().astype('int')[0] 
predictionLabels = [labels[i] for i in predictionLabels]
preictionBoxes = boxes.numpy()[0].astype('int')
finalScores = scores.numpy()[0]

for score, (ymin,xmin,ymax,xmax), label in zip(finalScores, preictionBoxes, predictionLabels):
    if score < 0.5:
        continue

    image_frame_boxed = cv2.rectangle(rgb,(xmin, ymax),(xmax, ymin),(215, 127, 255),2)      
    target_font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image_frame_boxed, label,(xmin, ymax-10), target_font, 1.5, (160,230,0), 2, cv2.LINE_AA)

plt.imshow(image_frame_boxed)
plt.show()