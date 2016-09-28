
#
# Script to create train/val datasets for automatic driver labelling
#

# Python
import sys
import json
import os
import shutil
# Numpy
import numpy as np


def usage():
    print "Usage : create_trainval_data.py /path/to/raw/data path/to/sloth/labels.json"

    
def get_annotations(filename):
    """
    :return: ndarray of dicts 
        {
            "annotations": [
                {
                    "class": "head",
                    "height": 150.0,
                    "type": "rect",
                    "width": 155.0,
                    "x": 52.0,
                    "y": 48.0
                },
                {
                    "class": "lhand",
                    "height": 51.0,
                    "type": "rect",
                    "width": 52.0,
                    "x": 420.0,
                    "y": 96.0
                },
                {
                    "class": "rhand",
                    "height": 73.0,
                    "type": "rect",
                    "width": 71.0,
                    "x": 458.0,
                    "y": 110.0
                },
                {
                    "class": "steer_wheel",
                    "height": 214.0,
                    "type": "rect",
                    "width": 115.0,
                    "x": 406.0,
                    "y": 99.0
                }],
            "class": "image",
            "filename": "train\\c0\\img_22590.jpg"
        }
    """
    labels = []
    with open(filename, 'r') as reader:
        str_data = ''.join(reader.readlines())
        raw_data = json.loads(str_data)
        for item in raw_data:
            if len(item['annotations']) > 0:
                labels.append(item)
    return np.array(labels)
   
def write_images_labels(annotations, data_path, output_path):
    
    output_images_folder = os.path.join(output_path, "images")
    output_labels_folder = os.path.join(output_path, "labels")
    os.makedirs(output_images_folder)
    os.makedirs(output_labels_folder)
        
    for i,annotation in enumerate(annotations):
        img_filename = annotation['filename']
        basename, ext = os.path.splitext(os.path.basename(img_filename))
        src_image_filename = os.path.join(data_path, img_filename)
        #dst_image_filename = os.path.join(output_images_folder, "%s%s" % (i,ext))
        dst_image_filename = os.path.join(output_images_folder, basename+ext)
        shutil.copyfile(src_image_filename, dst_image_filename)
        #dst_label_filename = os.path.join(output_labels_folder, "%s.txt" % i)
        dst_label_filename = os.path.join(output_labels_folder, basename + ".txt")
        with open(dst_label_filename, 'w') as writer:
            for obj in annotation['annotations']:
                # format : class_name bbox_left bbox_top bbox_right bbox_bottom
                l, t, w, h = int(obj['x']), int(obj['y']), int(obj['width']), int(obj['height'])
                line = "%s %s %s %s %s\n" % (obj['class'], l, t, l+w, t-h)
                writer.write(line)
        
if __name__ == "__main__":

    if len(sys.argv) != 3:
        usage()
        exit(1)

    RAW_DATA_PATH = sys.argv[1]
    SLOTH_LABELS_PATH = sys.argv[2]
    TRAIN_TEST_SPLIT=0.7
    
    annotations = get_annotations(SLOTH_LABELS_PATH)
    
    # Create data split
    num_labels = len(annotations)
    indices = np.random.permutation(num_labels)
    split_index = int(num_labels * TRAIN_TEST_SPLIT)
    train_annotations = annotations[indices[:split_index]]
    test_annotations = annotations[indices[split_index:]]
    
    print "Total : %s, Train : %s, Test : %s" % (num_labels, len(train_annotations), len(test_annotations))
    
    # Following DIGITS conventions : Label files are expected to have the .txt extension. 
    # For example if an image file is named foo.png the corresponding label file should be foo.txt. 
    # And specific for object detection
    # https://github.com/NVIDIA/DIGITS/tree/master/digits/extensions/data/objectDetection
    
    if os.path.isdir("train"):
        shutil.rmtree("train")
    if os.path.isdir("test"):
        shutil.rmtree("test")
    
    write_images_labels(train_annotations, RAW_DATA_PATH, "train")
    write_images_labels(test_annotations, RAW_DATA_PATH, "test")

