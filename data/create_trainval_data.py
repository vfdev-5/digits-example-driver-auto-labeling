
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
   
def write_images_labels(annotations, data_path, output_path, image_size):
    """ 
        LABEL STRUCTURE from DIGITS\digits\extensions\data\objectDetection\utils.py
    
        This class is the data ground-truth

        #Values    Name      Description
        ----------------------------------------------------------------------------
        1    type         Class ID
        1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                          truncated refers to the object leaving image boundaries.
                          -1 corresponds to a don't care region.
        1    occluded     Integer (-1,0,1,2) indicating occlusion state:
                          -1 = unkown, 0 = fully visible,
                          1 = partly occluded, 2 = largely occluded
        1    alpha        Observation angle of object, ranging [-pi..pi]
        4    bbox         2D bounding box of object in the image (0-based index):
                          contains left, top, right, bottom pixel coordinates
        3    dimensions   3D object dimensions: height, width, length (in meters)
        3    location     3D object location x,y,z in camera coordinates (in meters)
        1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
        1    score        Only for results: Float, indicating confidence in
                          detection, needed for p/r curves, higher is better.

        Here, 'DontCare' labels denote regions in which objects have not been labeled,
        for example because they have been too far away from the laser scanner.
    """

    
    output_images_folder = os.path.join(output_path, "images")
    output_labels_folder = os.path.join(output_path, "labels")
    os.makedirs(output_images_folder)
    os.makedirs(output_labels_folder)
    
    def _clamp(x, dim):
        return min(max(x, 0), dim-1)
        
    for i,annotation in enumerate(annotations):
        img_filename = annotation['filename']
        basename, ext = os.path.splitext(os.path.basename(img_filename))
        src_image_filename = os.path.join(data_path, img_filename)
        dst_image_filename = os.path.join(output_images_folder, "%s%s" % (i,ext))
        #dst_image_filename = os.path.join(output_images_folder, basename+ext)
        shutil.copyfile(src_image_filename, dst_image_filename)
        dst_label_filename = os.path.join(output_labels_folder, "%s.txt" % i)
        #dst_label_filename = os.path.join(output_labels_folder, basename + ".txt")
        with open(dst_label_filename, 'w') as writer:
            for obj in annotation['annotations']:
                # format : class_name bbox_left bbox_top bbox_right bbox_bottom
                l, t, w, h = int(obj['x']), int(obj['y']), int(obj['width']), int(obj['height'])
                r = l+w; b = t+h
                l = _clamp(l, image_size[0])
                t = _clamp(t, image_size[1])
                r = _clamp(r, image_size[0])
                b = _clamp(b, image_size[1])
                line = "{type} {truncated} {occluded} {alpha} {l} {t} {r} {b} {h} {w} {le} {x} {y} {z} {ry}\n".format(
                    type=obj['class'],
                    truncated=0.0,
                    occluded=-1,
                    alpha=0.0,
                    l=l, t=t, r=r, b=b,
                    h=0.0, w=0.0, le=0.0,
                    x=0.0, y=0.0, z=0.0,
                    ry = 0.0
                )   
                writer.write(line)
        
if __name__ == "__main__":

    if len(sys.argv) != 3:
        usage()
        exit(1)

    RAW_DATA_PATH = sys.argv[1]
    SLOTH_LABELS_PATH = sys.argv[2]
    TRAIN_TEST_SPLIT=0.8
    IMAGE_SIZE=[640, 480]
    
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
    
    write_images_labels(train_annotations, RAW_DATA_PATH, "train", IMAGE_SIZE)
    write_images_labels(test_annotations, RAW_DATA_PATH, "test", IMAGE_SIZE)
    
    print "Done."

