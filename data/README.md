# Training and testing data

We use a dataset from [Statefarm Kaggle competition](https://www.kaggle.com/c/state-farm-distracted-driver-detection).

Annotation is done using [sloth](https://github.com/cvhciKIT/sloth) application with a specific configuration.

### Labelling using Sloth

Assuming that Statefarm training dataset is at `~/sf-dataset/train` and this repository folder is at `~/repo/data`,
first time, run `sloth` with the configuration file `ddd_conf.py` :
```
$ cd ~/sf-dataset; sloth --config=~/repo/data/sloth-resources/ddd_conf.py 
```
When application opens you should see in the groupbox 'Labels': `head`, `rhand`, `lhand`, `steer_wheel`. Shortcuts to select these classes are :
* `z` - head
* `a` - left hand
* `d` - right hand
* `s` - steering wheel

To import image to label with configured labels, press `Ctrl+I` or in the menu `Edit -> Import Image` and select multiple images.
Be patient, it can take time to import images. Then you can start to label images. When you are done with labelling, save it. 
Application will create a `.json` file with the following structure :

```
[
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
            }
        ],
        "class": "image",
        "filename": "relative/path/to/image1.jpg"
    },
    ...
]    
```
Pay attention, that `filename` key has a relative path to the image file. 



### Create train/test datasets

Assuming that the labelling result is at `~/sf-dataset/labels.json`, run the script `create_trainval_data.py` from `~/repo/data`:
```
python create_trainval_data.py ~/sf-dataset/ ~/sf-dataset/labels_c0.json
```






