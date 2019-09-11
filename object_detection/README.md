# Poisonous Spider Object Detection
## Step 1: Intallation
Tensorflow Object Detection API depends on many libraries, so first we need to install all of them and set up the environment. Refer to [this](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) for more information.
## Step 2: Label images
Refer to [this](https://github.com/tzutalin/labelImg) for more information on how to use LabelImg to label images.
## Step 3: Convert to TFRecord file format
Tensorflow Object Detection API requires inputs in TFRecord file format. Thus, in order to use your own dataset, the data must be converted to .record format. Run twice convert_to_tfrecord.py file (once for the training set, once for the testing set), with the following command:
```bash
python convert_to_tfrecord.py 
	--output_path=${your output path} 
	--images_dir=${the path where the data are}
    --labels_dir=${the path where the xml files are}
```
Now, we have train.record and test.record files.





