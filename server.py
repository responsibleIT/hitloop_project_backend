######Pre-requisites#####
# General
import numpy as np
import string
from flask import Flask, request
from flask_cors import CORS
from pathlib import Path
import shutil
import json

# Model
import tensorflow as tf
from tensorflow.keras.models import load_model
import midi_functions
import os
from dotenv import load_dotenv


import random

load_dotenv()

##########################################General functions#####################################################
# directories
directory = str(Path(__file__).parent.absolute())

# import model
access_token = os.getenv('TOKEN')
file_name = 'generator.h5'
model_path = os.path.join(directory, file_name)


model = load_model(model_path)

###############App############################################
app = Flask(__name__)
CORS(app)


@app.route('/test_json', methods=['GET'])
#Requires input of 128 random numbers as input. In python used tf.random.normal([1,128])
def generate_midi():
    seed = int(request.args.get('seed'))
    noise = tf.random.normal([1, 128], seed=seed)

    generated_array = model(noise, training=False)

    # full_array = x.reshape(input_shape)
    new_json = midi_functions.convert_array2json(generated_array)

    return (new_json)


# @app.route('/test_json', methods=['GET'])
# def example_json():

#     midi_path = str(directory + '/' +'test_midi.json')
#     midi = open (midi_path, "r")

#     json_midi = json.load(midi)

#     return(json_midi)