# %% [markdown]
# # Introduction
# 
# In this notebook, we train [YOLOv4 tiny](https://github.com/AlexeyAB/darknet/issues/6067) on custom data. We will convert this to a TensorFlow representation and finally TensorFlow Lite file to use on device.
# 
# We also recommend reading our blog post on [How To Train YOLOv4 And Convert It To TensorFlow (And TensorFlow Lite!)](https://blog.roboflow.ai/how-to-train-yolov4-and-convert-it-to-tensorflow) side by side.
# 
# We will take the following steps to get YOLOv4 from training on custom data to a TensorFlow (and TensorFlow Lite) representation:
# 
# 
# 1.   Set up the Custom Dataset
# 2.   Train the Model with Darknet
# 3.   Convert the weights to TensorFlow's .pb representation
# 4.   Convert the weights to TensorFlow Lite's .tflite representation
# 
# 
# When you are done you will have a custom detector that you can use. It will make inference like this:
# 
# #### ![Roboflow Workmark](https://i.imgur.com/L0n564N.png)
# 
# ### **Reach out for support**
# 
# If you run into any hurdles on your own data set or just want to share some cool results in your own domain, [reach out!](https://roboflow.ai)
# 
# 
# 
# #### ![Roboflow Workmark](https://i.imgur.com/WHFqYSJ.png)

# %% [markdown]
# #1. Set up the Custom Dataset
# 
# 

# %% [markdown]
# We'll use Roboflow to convert our dataset from any format to the YOLO Darknet format.
# 
# 1. To do so, create a free [Roboflow account](https://app.roboflow.ai).
# 2. Upload your images and their annotations (in any format: VOC XML, COCO JSON, TensorFlow CSV, etc).
# 3. Apply preprocessing and augmentation steps you may like. We recommend at least `auto-orient` and a `resize` to 416x416. Generate your dataset.
# 4. Export your dataset in the **YOLO Darknet format**.
# 5. Copy your download link, and paste it below.
# 
# See our [blog post](https://blog.roboflow.ai/training-yolov4-on-a-custom-dataset/) for greater detail.
# 
# In this example, I used the open source [BCCD Dataset](https://public.roboflow.ai/object-detection/bccd). (You can `fork` it to your Roboflow account to follow along.)

# %%
#if you already have YOLO darknet format, you can skip this step
#otherwise we recommend formatting in Roboflow
%cd /content
%mkdir dataset
%cd ./dataset
!curl -L "https://universe.roboflow.com/ds/UPgPU3FL30?key=im8oKcDNtQ" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip

# %% [markdown]
# #2. Train a Custom Model on DarkNet

# %% [markdown]
# ***Since we already have a [notebook](https://colab.research.google.com/drive/1PWOwg038EOGNddf6SXDG5AsC8PIcAe-G#scrollTo=NjKzw2TvZrOQ) on how to train YOLOv4 with Darknet, we have simply included the contents here as well.***

# %% [markdown]
# ## Introduction
# 
# 
# In this notebook, we implement the tiny version of [YOLOv4](https://arxiv.org/pdf/2004.10934.pdf) for training on your own dataset, [YOLOv4 tiny](https://github.com/AlexeyAB/darknet/issues/6067).
# 
# We also recommend reading our blog post on [Training YOLOv4 on custom data](https://blog.roboflow.ai/training-yolov4-on-a-custom-dataset/) side by side.
# 
# We will take the following steps to implement YOLOv4 on our custom data:
# * Configure our GPU environment on Google Colab
# * Install the Darknet YOLOv4 training environment
# * Download our custom dataset for YOLOv4 and set up directories
# * Configure a custom YOLOv4 training config file for Darknet
# * Train our custom YOLOv4 object detector
# * Reload YOLOv4 trained weights and make inference on test images
# 
# When you are done you will have a custom detector that you can use. It will make inference like this:
# 
# #### ![Roboflow Workmark](https://i.imgur.com/L0n564N.png)
# 
# ### **Reach out for support**
# 
# If you run into any hurdles on your own data set or just want to share some cool results in your own domain, [reach out!](https://roboflow.ai)
# 
# 
# 
# #### ![Roboflow Workmark](https://i.imgur.com/WHFqYSJ.png)

# %% [markdown]
# ## Configuring CUDA on Colab for YOLOv4
# 
# 

# %%
# CUDA: Let's check that Nvidia CUDA drivers are already pre-installed and which version is it. This can be helpful for debugging.
!/usr/local/cuda/bin/nvcc --version

# %% [markdown]
# **IMPORTANT!** If you're not training on a Tesla P100 GPU, we will need to tweak our Darknet configuration later based on what type of GPU we have. Let's set that now while we're inspecting the GPU.

# %%
#take a look at the kind of GPU we have
!nvidia-smi

# %%
import tensorflow as tf
print("TensorFlow version:", tf.__version__)

# Check if GPU is available
print("GPU Available:", tf.config.list_physical_devices('GPU'))


# %%
# Change the number depending on what GPU is listed above, under NVIDIA-SMI > Name.
# Tesla K80: 30
# Tesla P100: 60
# Tesla T4: 75
%env compute_capability=75

# %% [markdown]
# ## Installing Darknet for YOLOv4 on Colab
# 
# 
# 

# %%
%cd /content/
%rm -rf darknet

# %%
#we clone the fork of darknet maintained by roboflow
#small changes have been made to configure darknet for training
!git clone https://github.com/AlexeyAB/darknet

# %% [markdown]
# **IMPORTANT! If you're not using a Tesla P100 GPU**, then uncomment the sed command and replace the arch and code with that matching your GPU. A list can be found [here](http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/). You can check with the command nvidia-smi (should be run above).

# %%
#install environment from the Makefile. Changes to mitigate CUDA error.

# !sed -i 's/OPENCV=0/OPENCV=1/g' Makefile
# !sed -i 's/GPU=0/GPU=1/g' Makefile
# !sed -i 's/CUDNN=0/CUDNN=1/g' Makefile
# !sed -i "s/ARCH= -gencode arch=compute_60,code=sm_60/ARCH= -gencode arch=compute_${compute_capability},code=sm_${compute_capability}/g" Makefile
# !make


%cd /content/darknet/
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile
!sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
!sed -i 's/LIBSO=0/LIBSO=1/' Makefile


# %%
!make

# %%
#download the newly released yolov4-tiny weights
%cd /content/darknet
!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29

# %%


# %% [markdown]
# ## Configure from Custom Dataset

# %%
#Copy dataset
%cp -r /content/dataset/. /content/darknet/
#Set up training file directories for custom dataset
%cd /content/darknet/
%cp train/_darknet.labels data/obj.names
%mkdir data/obj
#copy image and labels
%cp train/*.jpg data/obj/
%cp valid/*.jpg data/obj/

%cp train/*.txt data/obj/
%cp valid/*.txt data/obj/

with open('data/obj.data', 'w') as out:
  out.write('classes = 3\n')
  out.write('train = data/train.txt\n')
  out.write('valid = data/valid.txt\n')
  out.write('names = data/obj.names\n')
  out.write('backup = backup/')

#write train file (just the image list)
import os

with open('data/train.txt', 'w') as out:
  for img in [f for f in os.listdir('train') if f.endswith('jpg')]:
    out.write('data/obj/' + img + '\n')

#write the valid file (just the image list)
import os

with open('data/valid.txt', 'w') as out:
  for img in [f for f in os.listdir('valid') if f.endswith('jpg')]:
    out.write('data/obj/' + img + '\n')

# %% [markdown]
# ## Write Custom Training Config for YOLOv4

# %%
#we build config dynamically based on number of classes
#we build iteratively from base config files. This is the same file shape as cfg/yolo-obj.cfg
def file_len(fname):
  with open(fname) as f:
    for i, l in enumerate(f):
      pass
  return i + 1

num_classes = file_len('train/_darknet.labels')
max_batches = num_classes*10
steps1 = .8 * max_batches
steps2 = .9 * max_batches
steps_str = str(steps1)+','+str(steps2)
num_filters = (num_classes + 5) * 3


print("writing config for a custom YOLOv4 detector detecting number of classes: " + str(num_classes))

#Instructions from the darknet repo
#change line max_batches to (classes*2000 but not less than number of training images, and not less than 6000), f.e. max_batches=6000 if you train for 3 classes
#change line steps to 80% and 90% of max_batches, f.e. steps=4800,5400
if os.path.exists('./cfg/custom-yolov4-tiny-detector.cfg'): os.remove('./cfg/custom-yolov4-tiny-detector.cfg')


#customize iPython writefile so we can write variables
from IPython.core.magic import register_line_cell_magic

@register_line_cell_magic
def writetemplate(line, cell):
    with open(line, 'w') as f:
        f.write(cell.format(**globals()))





# %%
%%writetemplate ./cfg/custom-yolov4-tiny-detector.cfg
[net]
# Testing
#batch=1
#subdivisions=1
# Training
batch=32
subdivisions=16
width=416
height=416
channels=3
momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1

learning_rate=0.00261
burn_in=1000
max_batches = {max_batches}
policy=steps
steps={steps_str}
scales=.1,.1

[convolutional]
batch_normalize=1
filters=32
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[route]
layers=-1
groups=2
group_id=1

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky

[route]
layers = -1,-2

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

[route]
layers = -6,-1

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[route]
layers=-1
groups=2
group_id=1

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[route]
layers = -1,-2

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[route]
layers = -6,-1

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[route]
layers=-1
groups=2
group_id=1

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[route]
layers = -1,-2

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[route]
layers = -6,-1

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

##################################

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters={num_filters}
activation=linear



[yolo]
mask = 3,4,5
anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
classes={num_classes}
num=6
jitter=.3
scale_x_y = 1.05
cls_normalizer=1.0
iou_normalizer=0.07
iou_loss=ciou
ignore_thresh = .7
truth_thresh = 1
random=0
nms_kind=greedynms
beta_nms=0.6

[route]
layers = -4

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[upsample]
stride=2

[route]
layers = -1, 23

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters={num_filters}
activation=linear

[yolo]
mask = 1,2,3
anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
classes={num_classes}
num=6
jitter=.3
scale_x_y = 1.05
cls_normalizer=1.0
iou_normalizer=0.07
iou_loss=ciou
ignore_thresh = .7
truth_thresh = 1
random=0
nms_kind=greedynms
beta_nms=0.6

# %%


# %%


# %%
#here is the file that was just written.
#you may consider adjusting certain things

#like the number of subdivisions 64 runs faster but Colab GPU may not be big enough
#if Colab GPU memory is too small, you will need to adjust subdivisions to 16
%cat cfg/custom-yolov4-tiny-detector.cfg

# %%
!cp '/content/multi_class_frame_yolov4tiny_best.weights' '/content/darknet/backup'

# %% [markdown]
# ## Train Custom YOLOv4 Detector

# %%
#use the below code line for default training
!./darknet detector train data/obj.data cfg/custom-yolov4-tiny-detector.cfg yolov4-tiny.conv.29 -dont_show -map
#If you get CUDA out of memory adjust subdivisions above!
#Use below code line to leverage transfer learning on yolov4tiny
!./darknet detector train data/obj.data cfg/custom-yolov4-tiny-detector.cfg backup/multi_class_frame_yolov4tiny_best.weights -clear

# %%


# %%


# %% [markdown]
# ## Infer Custom Objects with Saved YOLOv4 Weights

# %%
#define utility function
def imShow(path):
  import cv2
  import matplotlib.pyplot as plt
  %matplotlib inline

  image = cv2.imread(path)
  height, width = image.shape[:2]
  resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)

  fig = plt.gcf()
  fig.set_size_inches(18, 10)
  plt.axis("off")
  #plt.rcParams['figure.figsize'] = [10, 5]
  plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
  plt.show()

# %%
#check if weigths have saved yet
#backup houses the last weights for our detector
#(file yolo-obj_last.weights will be saved to the build\darknet\x64\backup\ for each 100 iterations)
#(file yolo-obj_xxxx.weights will be saved to the build\darknet\x64\backup\ for each 1000 iterations)
#After training is complete - get result yolo-obj_final.weights from path build\darknet\x64\bac
!ls backup
#if it is empty you haven't trained for long enough yet, you need to train for at least 100 iterations

# %%
#save final weights to google drive
from google.colab import drive
drive.mount('/content/drive')

# %%
# Darknet Weights
!cp /content/darknet/backup/custom-yolov4-tiny-detector_10000.weights "/content/drive/My Drive"

# %%
#coco.names is hardcoded somewhere in the detector
%cp data/obj.names data/coco.names

# %%

# /test has images that we can test our detector on
test_images = [f for f in os.listdir('test') if f.endswith('.jpg')]
import random
img_path = "test/" + random.choice(test_images);
# img_path='/content/darknet/train/08976546789_jpg.rf.4203c8a46792be6971477dadc0cec634.jpg'
#test out our detector!
!./darknet detect cfg/custom-yolov4-tiny-detector.cfg backup/custom-yolov4-tiny-detector_last.weights {img_path} -dont-show
imShow('predictions.jpg')

# %% [markdown]
# #3. Convert the weights to TensorFlow's .pb representation

# %% [markdown]
# Darknet produces a .weights file specific to Darknet. If we want to use the YOLOv4 model in TensorFlow, we'll need to convert it.
# 
# To do this, we'll use the following tool: https://github.com/hunglc007/tensorflow-yolov4-tflite.

# %% [markdown]
# ## Install and Configure

# %% [markdown]
# First, we'll clone the repository.

# %%
%cd /content
!git clone https://github.com/hunglc007/tensorflow-yolov4-tflite.git
%cd /content/tensorflow-yolov4-tflite

# %% [markdown]
# Then, we'll change the labels from the default COCO to our own custom ones.

# %%
!cp /content/darknet/data/obj.names /content/tensorflow-yolov4-tflite/data/classes/
!ls /content/tensorflow-yolov4-tflite/data/classes/

# %%
!sed -i "s/coco.names/obj.names/g" /content/tensorflow-yolov4-tflite/core/config.py

# %% [markdown]
# ## Convert

# %% [markdown]
# Time to convert! We'll convert to both a regular TensorFlow SavedModel and to TensorFlow Lite. For TensorFlow Lite, we'll convert to a different TensorFlow SavedModel beforehand.

# %%
%cd /content/tensorflow-yolov4-tflite
# Regular TensorFlow SavedModel
!python save_model.py \
  --weights /content/darknet/backup/custom-yolov4-tiny-detector_last.weights \
  --output ./checkpoints/yolov4-tiny-416 \
  --input_size 416 \
  --model yolov4 \
  --tiny \

# SavedModel to convert to TFLite
!python save_model.py \
  --weights /content/darknet/backup/custom-yolov4-tiny-detector_last.weights \
  --output ./checkpoints/yolov4-tiny-pretflite-416 \
  --input_size 416 \
  --model yolov4 \
  --tiny \
  --framework tflite

# %%


# %%


# %% [markdown]
# #4. Convert the TensorFlow weights to TensorFlow Lite

# %% [markdown]
# From the generated TensorFlow SavedModel, we will convert to .tflite

# %%
# %cd /content/tensorflow-yolov4-tflite
# !python convert_tflite.py --weights ./checkpoints/yolov4-tiny-pretflite-416 --output ./checkpoints/yolov4-tiny-416.tflite


# %%
import tensorflow as tf

# Load the SavedModel
saved_model_dir = "/content/tensorflow-yolov4-tflite/checkpoints/yolov4-tiny-416"
model = tf.saved_model.load(saved_model_dir)

# Define a function to reshape the output
def reshape_output(x):
    return tf.reshape(x, [1, 2535, 4])

# Get the concrete function for inference
concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

# Wrap the concrete function with the reshape_output function
@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
def wrapped_concrete_func(x):
    output = concrete_func(x)
    return reshape_output(output['output'])  # Assuming 'output' is the key for your output tensor

# Convert the wrapped concrete function to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_concrete_functions([wrapped_concrete_func])
tflite_model = converter.convert()

# Save the converted model
with open("converted_model.tflite", "wb") as f:
    f.write(tflite_model)



# %%
import numpy as np
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print input details
print("Input shape:", input_details[0]['shape'])
print("Input type:", input_details[0]['dtype'])

# Print output details
print("Output shape:", output_details[0]['shape'])
print("Output type:", output_details[0]['dtype'])


# %%
!ls /content/darknet/test

# %%
# Verify
%cd /content/tensorflow-yolov4-tflite
!python detect.py --weights ./checkpoints/yolov4-tiny-416 --size 416 --model yolov4 \
  --image  '/content/darknet/test/15874452230_ba58ac5841_b_jpg.rf.ebfca4006964db02c348e158531f2df3.jpg'
  # --framework tflite

# %%
%cd /content/tensorflow-yolov4-tflite/
!ls
from IPython.display import Image
Image('/content/tensorflow-yolov4-tflite/result.png')

# %% [markdown]
# # Save your Model

# %% [markdown]
# You can save your model to your Google Drive for further use.

# %%
# Choose what to copy

# TensorFlow SavedModel
!cp -r /content/tensorflow-yolov4-tflite/checkpoints/yolov4-tiny-416/ "/content/drive/My Drive"
# TensorFlow Lite
!cp /content/tensorflow-yolov4-tflite/checkpoints/yolov4-tiny-416.tflite "/content/drive/My Drive"

# %%
import tensorflow as tf
import numpy as np

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="your_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print input and output details
print("Input details:", input_details)
print("Output details:", output_details)

# Modify output shape (assuming single output tensor)
output_shape = (1, 10)  # Change output shape as needed
interpreter.set_tensor(output_details[0]['index'], np.zeros(output_shape, dtype=output_details[0]['dtype']))

# Run inference
interpreter.invoke()

# Get the modified output
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Modified output shape:", output_data.shape)



