######Pre-requisites#####
# General
import numpy as np
import string
from flask import Flask, request, render_template, jsonify, send_file
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


# Front page
@app.route('/', methods=['GET'])
def index():
    return render_template('info.html')


# Midi generation page
@app.route('/midi_json', methods=['GET'])
# Requires input of 128 random numbers as input. In python used tf.random.normal([1,128])
def generate_midi():
    seed = int(request.args.get('seed'))
    tf.random.set_seed(seed)
    noise = tf.random.normal([1, 128])

    generated_array = model(noise, training=False)
    np_array = np.array(generated_array).reshape(128, 20)

    # Filter out too low values. Because current model is chance-based all values should be above 0.9. Everythin below changed to 0
    np_array[np_array < 0.9] = 0
    np_array[np_array > 1] = 1

    # full_array = x.reshape(input_shape)
    new_json = midi_functions.convert_array2json(np_array)

    return (new_json)


@app.route('/midi_compressed_json', methods=['GET'])
# Requires input of 128 random numbers as input. In python used tf.random.normal([1,128])
def generate_base_midi():
    seed = int(request.args.get('seed'))
    tf.random.set_seed(seed)
    noise = tf.random.normal([1, 128])

    generated_array = model(noise, training=False)
    np_array = np.array(generated_array).reshape(128, 20)

    # Filter out too low values. Because current model is chance-based all values should be above 0.9. Everythin below changed to 0
    np_array[np_array < 0.9] = 0
    np_array[np_array > 1] = 1

    np_array_compressed = midi_functions.compress_bass(np_array)

    # full_array = x.reshape(input_shape)
    new_json = midi_functions.convert_array2json(np_array_compressed)

    return (new_json)



@app.route('/sequencer_json', methods=['GET'])
# Requires input of 128 random numbers as input. In python used tf.random.normal([1,128])
def generate_sequencer_json():
    seed = int(request.args.get('seed'))
    tf.random.set_seed(seed)
    noise = tf.random.normal([1, 128])

    generated_array = model(noise, training=False)
    np_array = np.array(generated_array).reshape(128, 20)

    # Filter out too low values. Because current model is chance-based all values should be above 0.9. Everythin below changed to 0
    np_array[np_array < 0.9] = 0
    np_array[np_array > 1] = 1

    np_array_compressed = midi_functions.compress_bass(np_array)

    np_array_final = midi_functions.resize_ouput_length(np_array_compressed, output_length=16)

    new_json = json.dumps(np_array_final.tolist())

    return (new_json)

############### Samples part ###############

@app.route('/samples_list', methods=['GET'])
def generate_samples_list():
    samples_folder_path = os.path.join(directory, 'samples')
    samples_available = os.listdir(samples_folder_path)
    return jsonify(files=samples_available)


@app.route('/samples', methods=['GET'])
def get_sample():
    file_name = str(request.args.get('file'))
    samples_folder_path = os.path.join(directory, 'samples')
    audio_path = os.path.join(samples_folder_path, file_name)
    return send_file(audio_path)