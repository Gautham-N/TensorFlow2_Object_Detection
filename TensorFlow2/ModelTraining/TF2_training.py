from google.colab import drive
drive.mount('/content/gdrive')

import tensorflow as tf
tf.test.gpu_device_name()

#cd into the TensorFlow directory in your Google Drive
%cd '/content/gdrive/My Drive/Tensorflow'

#cloning Tensoflow model.git
!git clone https://github.com/tensorflow/models.git

%cd '/content/gdrive/MyDrive/Tensorflow/models'
!git checkout -f e04dafd04d69053d3733bb91d47d0d95bc2c8199

#Installing required libararies
!apt-get install protobuf-compiler python-lxml python-pil
!pip install Cython pandas tf-slim lvis

#cd into 'TensorFlow/models/research'
%cd '/content/gdrive/My Drive/Tensorflow/models/research/'

!protoc object_detection/protos/*.proto --python_out=.

import os
import sys
os.environ['PYTHONPATH']+=":/content/gdrive/My Drive/Tensorflow/models"
sys.path.append("/content/gdrive/My Drive/Tensorflow/models/research")

!python setup.py build
!python setup.py install

cd '/content/gdrive/My Drive/Tensorflow/models/research/object_detection/builders/'

#Building model_builder_tf2
!python model_builder_tf2_test.py
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
print('Done')

%cd '/content/gdrive/My Drive/Tensorflow/scripts/preprocessing'

#Generation tfrecord for train set
!python generate_tfrecord.py -x '/content/gdrive/My Drive/Tensorflow/workspace/training_demo/annotations/fruite/train_zip/train' -l '/content/gdrive/My Drive/Tensorflow/workspace/training_demo/annotations/rec/frt.pbtxt' -o '/content/gdrive/My Drive/Tensorflow/workspace/training_demo/annotations/fruite/train.record'

#Generation tfrecord for test set
!python generate_tfrecord.py -x '/content/gdrive/My Drive/Tensorflow/workspace/training_demo/annotations/fruite/test' -l '/content/gdrive/My Drive/Tensorflow/workspace/training_demo/annotations/rec/frt.pbtxt' -o '/content/gdrive/My Drive/Tensorflow/workspace/training_demo/annotations/fruite/test.record'

#cd into training_demo
%cd '/content/gdrive/My Drive/Tensorflow/workspace/training_demo'

#Training the model
!python model_main_tf2.py --model_dir='/content/gdrive/My Drive/Tensorflow/FRCNNFruite300/' --pipeline_config_path='/content/gdrive/My Drive/Tensorflow/workspace/training_demo/models/FRCNN/pipeline.config'

#Exporting as saved_model
!python exporter_main_v2.py --input_type image_tensor --pipeline_config_path '/content/gdrive/My Drive/Tensorflow/workspace/training_demo/models/FRCNN/pipeline.config' --trained_checkpoint_dir '/content/gdrive/My Drive/Tensorflow/FRCNNFruite300' --output_directory '/content/gdrive/My Drive/Tensorflow/FRCNNFruiteSavedM/'


