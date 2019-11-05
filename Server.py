#-*-coding:utf-8-*-
from flask import Flask
import werkzeug
from flask import request
import os
import cv2
import numpy as np
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# for spider recognition
import recognition






app = Flask(__name__)
basedir=os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN']=True
app.config['SQLALCHEMY_TRACK_MODIFICATIONS']=True

# ------------------ spider Model Initialization ------------------------------ #
spider_label_map = label_map_util.load_labelmap('/home/google/model/object-detection/labelmap.pbtxt')
spider_categories = label_map_util.convert_label_map_to_categories(
    spider_label_map, max_num_classes=11, use_display_name=True)
spider_category_index = label_map_util.create_category_index(spider_categories)

spider_detection_graph = tf.Graph()

with spider_detection_graph.as_default():
    spider_od_graph_def = tf.GraphDef()
    with tf.gfile.GFile('/home/google/model/object-detection/spider.pb', 'rb') as fid:
        spider_serialized_graph = fid.read()
        spider_od_graph_def.ParseFromString(spider_serialized_graph)
        tf.import_graph_def(spider_od_graph_def, name='')

    spider_session = tf.Session(graph=spider_detection_graph)

spider_image_tensor = spider_detection_graph.get_tensor_by_name(
    'image_tensor:0')
spider_detection_boxes = spider_detection_graph.get_tensor_by_name(
    'detection_boxes:0')
spider_detection_scores = spider_detection_graph.get_tensor_by_name(
    'detection_scores:0')
spider_detection_classes = spider_detection_graph.get_tensor_by_name(
    'detection_classes:0')
spider_num_detections = spider_detection_graph.get_tensor_by_name(
    'num_detections:0')
# ---------------------------------------------------------------------------- #
# ------------------ General Model Initialization ---------------------------- #
# What model to download.
MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = '/home/google/model/object-detection/general.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '/home/google/model/object-detection/mscoco_label_map.pbtxt'
general_label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
general_categories = label_map_util.convert_label_map_to_categories(
    general_label_map, max_num_classes=90, use_display_name=True)
general_category_index = label_map_util.create_category_index(
    general_categories)

general_detection_graph = tf.Graph()

with general_detection_graph.as_default():
    general_od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        general_serialized_graph = fid.read()
        general_od_graph_def.ParseFromString(general_serialized_graph)
        tf.import_graph_def(general_od_graph_def, name='')

    general_session = tf.Session(graph=general_detection_graph)

general_image_tensor = general_detection_graph.get_tensor_by_name(
    'image_tensor:0')
general_detection_boxes = general_detection_graph.get_tensor_by_name(
    'detection_boxes:0')
general_detection_scores = general_detection_graph.get_tensor_by_name(
    'detection_scores:0')
general_detection_classes = general_detection_graph.get_tensor_by_name(
    'detection_classes:0')
general_num_detections = general_detection_graph.get_tensor_by_name(
    'num_detections:0')
# ---------------------------------------------------------------------------- #




def spider(image_path):
    try:
        image = cv2.imread(image_path)
        image_expanded = np.expand_dims(image, axis=0)
        (boxes, scores, classes, num) = spider_session.run(
            [spider_detection_boxes, spider_detection_scores,
                spider_detection_classes, spider_num_detections],
            feed_dict={spider_image_tensor: image_expanded})
        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)
        boxes = np.squeeze(boxes)
        object_name = []
        object_score = []

        for c in range(0, len(classes)):
            class_name = spider_category_index[classes[c]]['name']
            if scores[c] > .80:
                object_name.append(class_name)
                object_score.append(str(scores[c] * 100)[:5])
    except:
        print("Error occurred in spider detection")
        object_name = ['']
        object_score = ['']
    return object_name, object_score

def general(image_path):
    try:
        image = cv2.imread(image_path)
        image_expanded = np.expand_dims(image, axis=0)
        (boxes, scores, classes, num) = general_session.run(
            [general_detection_boxes, general_detection_scores,
                general_detection_classes, general_num_detections],
            feed_dict={general_image_tensor: image_expanded})

        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)
        boxes = np.squeeze(boxes)

        object_name = []
        object_score = []

        for c in range(0, len(classes)):
            class_name = general_category_index[classes[c]]['name']
            if scores[c] > .30:   # If confidence level is good enough
                object_name.append(class_name)
                object_score.append(str(scores[c] * 100)[:5])
    except:
        print("Error occurred in general detection")
        object_name = ['']
        object_score = ['']

    return object_name, object_score

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(
                    tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(
                    tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(
                    tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [
                                           real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                                           real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[1], image.shape[2])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: image})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(
                output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

@app.route('/')
def test():
    return '服务器正常运行'





#此方法处理用户注册
@app.route('/spiders',methods=['POST'])
def spiders():
    imagefile =request.files['image']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + filename)
    image_path = "/home/google/test_image/" + filename
    imagefile.save(image_path)

    result_category = ''
    sname, sscores = spider(image_path)  # to detect whether spider is present in image(this is custom trained model)
    gname, gscores = general(image_path)  # to detect those 90 objects from TF API
    print(sscores)
    print(gscores)
    count = 0
    if (len(gscores) == 0 or len(gscores[0])==0) and (len(sscores) >0 or len(sscores[0])>0 ):
        result_category = 'spider'
        for i in sscores:
            if float(i) > 95:
                count += 1
    elif (len(sscores) == 0 or len(sscores[0])==0) and (len(gscores)>0 or len(gscores[0])>0):
        result_category = gname[0]
        for i in sscores:
            if float(i) > 95:
                count += 1
    elif (len(sscores) == 0 or len(sscores[0])==0)and (len(gscores) == 0 or len(gscores[0])==0):
        print('cannot detect')
    else:
        if float(sscores[0]) > float(gscores[0]):
            result_category = 'spider'
        else:
            result_category = gname[0]
        for i in sscores:
            if float(i) > 95:
                count += 1
        for i in gscores:
            if float(i) > 95:
                count += 1
    print(result_category)    
    if count > 1:
        print("have multiple objects")
        result_category = "multiple"
        os.remove(image_path)
        return result_category
    else: 
        if (result_category == 'spider'):
            result = recognition.classify(image_path)
            os.remove(image_path)
            return result
        else:
            os.remove(image_path)
            return result_category

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)
