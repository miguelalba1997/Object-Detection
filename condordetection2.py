import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import cv2

from utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util

cap = cv2.VideoCapture(0)

sys.path.append("..")

MODEL_NAME = 'C:/Users/migue/Documents/CondorDetection/inference_graph'

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

PATH_TO_LABELS = os.path.join('C:/Users/migue/Documents/CondorDetection/object_detection/training', 'labelmap.pbtxt')

NUM_CLASSES = 1

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)

categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes = NUM_CLASSES, 
                                                            use_display_name = True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

PATH_TO_TEST_IMAGES_DIR = 'images/test'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'img{}.jpg'.format(i)) for i in range(1, 10) ]
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

#def run_inference_for_single_image(image, graph):
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:
        #for image_path in TEST_IMAGE_PATHS:
            #image = Image.open(image_path)
            
            #image_np = load_image_into_numpy_array(image)
            
            ret, image_np = cap.read()
            
            image_np_expanded = np.expand_dims(image_np, axis = 0)
            
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections], 
                feed_dict={image_tensor: image_np_expanded})
            
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates = True,
                line_thickness=8)
            #plt.figure(figsize=IMAGE_SIZE)
            #plt.imshow(image_np)
            
            cv2.imshow('Condor Live Feed', cv2.resize(image_np,(800,600)))
            cv2.waitKey(1)
            if cv2.getWindowProperty('Condor Live Feed', 1) < 0:
                break
        cap.release()    
        cv2.destroyWindow('Condor Live Feed')
