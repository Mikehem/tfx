import os
import sys
import datetime

BASE_PATH = "/home/MD00560695/workdir/tfx/"
sys.path.append(BASE_PATH)
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

# DATA PATHS
DATA_PATH = os.path.join("/home/MD00560695/workdir/building-machine-learning-pipelines/", "data")
COMPLAINTS_DATA_PATH = os.path.join(DATA_PATH, "US_consumer_complaints")
COMPLAINTS_TF_DATA_PATH = os.path.join(COMPLAINTS_DATA_PATH, "tfrecords")
COMPLAINTS_RAW_TF_DATA_PATH = os.path.join(COMPLAINTS_TF_DATA_PATH, "raw")
COMPLAINTS_SPLIT_TF_DATA_PATH = os.path.join(COMPLAINTS_TF_DATA_PATH, "split")
complaints_tf_file = os.path.join(COMPLAINTS_RAW_TF_DATA_PATH, "consumer_complaints.tfrecord")
complaints_tf_val_file  = os.path.join(COMPLAINTS_SPLIT_TF_DATA_PATH, "val", "consumer_complaints_val.tfrecord")
complaints_tf_test_file  = os.path.join(COMPLAINTS_SPLIT_TF_DATA_PATH, "test","consumer_complaints_test.tfrecord")
complaints_tf_train_file  = os.path.join(COMPLAINTS_SPLIT_TF_DATA_PATH, "train","consumer_complaints_train.tfrecord")

# TFX configurations
_pipeline_name = 'consumer_complaints'

# This example assumes that the taxi data is stored in ~/taxi/data and the
# taxi utility function is in ~/taxi.  Feel free to customize this as needed.
# os.environ['HOME']
_consumer_complaints_root = os.path.join(BASE_PATH, "tfx/examples", 'consumer_complaints')
_data_root = os.path.join(complaints_tf_train_file)
# Python module file to inject customized logic into the TFX components. The
# Transform and Trainer both require user-defined functions to run successfully.
_module_file = os.path.join(_consumer_complaints_root, 'consumer_complaints_utils.py')
# Path which can be listened to by the model server.  Pusher will output the
# trained model here.
_serving_model_dir = os.path.join(_consumer_complaints_root, 'serving_model', _pipeline_name)

# Directory and data locations.  This example assumes all of the chicago taxi
# example code and metadata library is relative to $HOME, but you can store
# these files anywhere on your local filesystem.
_tfx_root = os.path.join(BASE_PATH, "tfx/examples", "consumer_complaints")
_tfx_pipeline = os.path.join(_tfx_root, 'pipelines', _pipeline_name)
_tfx_modules = os.path.join(_tfx_root, "modules")
# Sqlite ML-metadata db path.
_tfx_metadata_path = os.path.join(_tfx_root, 'metadata', _pipeline_name,
                              'metadata.db')

# Pipeline arguments for Beam powered Components.
_beam_pipeline_args = [
    '--direct_running_mode=multi_processing',
    # 0 means auto-detect based on on the number of CPUs available
    # during execution time.
    '--direct_num_workers=0',
]

# Airflow-specific configs; these will be passed directly to airflow
_airflow_config = {
    'schedule_interval': None,
    'start_date': datetime.datetime(2020, 11, 1),
}

# imports
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
from tfx.components import CsvExampleGen
from tfx.proto import example_gen_pb2
import tensorflow as tf

import pandas as pd
import csv
import re 

# Set the Interaction Context
context = InteractiveContext(pipeline_root= _tfx_pipeline)

# Ingesting the data
from tfx.components import CsvExampleGen
from tfx.components import ImportExampleGen
from tfx.utils.dsl_utils import external_input
from tfx.proto import example_gen_pb2
import tensorflow_data_validation as tfdv

train_stats = tfdv.generate_statistics_from_tfrecord(data_location=complaints_tf_train_file)
val_stats = tfdv.generate_statistics_from_tfrecord(data_location=complaints_tf_val_file)

print(train_stats)

schema = tfdv.infer_schema(train_stats)

val_anomalies = tfdv.validate_statistics(statistics=val_stats, schema=schema)

print(val_anomalies)