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
import mido
import pickle
from sklearn.neural_network import BernoulliRBM
from huggingface_hub import hf_hub_download
import midi_functions
import os
from dotenv import load_dotenv

load_dotenv()

##########################################General functions#####################################################
# directories
directory = str(Path(__file__).parent.absolute())

# import model

access_token = os.getenv('TOKEN')
file_name = '/model.pkl'
model_path = str(directory + file_name)

if os.path.exists(model_path) == False:
    hf_hub_download(repo_id=os.getenv('REPO_ID'), filename="model.pkl",
                    use_auth_token=access_token, cache_dir=directory)
    snapshots_folder = os.getenv('SNAPSHOTS_FOLDER')
    folder = os.listdir(str(directory + snapshots_folder))[0]
    path_to_downloaded_model = str(directory + snapshots_folder + '/' + folder + file_name)
    shutil.copy(path_to_downloaded_model, model_path)
    shutil.rmtree(str(directory + '/' + snapshots_folder))

midi_songs_path = str(directory + '/songs')
input_midi_folders = os.listdir(midi_songs_path)

used_files = []
midi_dataset = []
for folder in input_midi_folders:
    loc = str(midi_songs_path + '/' + folder)
    temp_midi = midi_functions.convert_midisong_2array(loc)
    if np.isnan(temp_midi).any():
        print('Contains Nan, dropping from dataset')
    else:
        midi_dataset.append(temp_midi)
        used_files.append(folder)

midi_drum_dataset = []
drum_lookup = midi_functions.drum_dict()
for item in midi_dataset:
    new_array = item[:, :len(drum_lookup)]
    midi_drum_dataset.append(new_array)

# Initialize model
with open(str(directory + file_name), 'rb') as file:
    RBMs = pickle.load(file)

###############App############################################
app = Flask(__name__)
CORS(app)


@app.route('/test_json', methods=['GET'])
def generate_midi():
    seed = int(request.args.get('seed'))
    template_midi = int(request.args.get('template'))

    # Prepare input data
    midi_sample = midi_drum_dataset[template_midi]
    input_shape = np.shape(midi_sample)
    flatten_size = input_shape[0] * input_shape[1]
    x = midi_sample.reshape((flatten_size)).astype("float32")

    # Run model
    rng = np.random.RandomState(seed)
    for rbm in RBMs:
        x = rbm._sample_hiddens(x, rng)
    for rbm in reversed(RBMs):
        x = rbm._sample_visibles(x, rng)

    # create new_shape for array due to only having drums
    empty_array = np.empty((2048, 128))
    new_array = x.reshape(input_shape)

    full_array = np.concatenate((new_array, empty_array, empty_array), axis=1)

    # full_array = x.reshape(input_shape)
    new_json = midi_functions.convert_array2json(full_array, tick_per_beat= 4)

    return (new_json)


# @app.route('/test_json', methods=['GET'])
# def example_json():

#     midi_path = str(directory + '/' +'test_midi.json')
#     midi = open (midi_path, "r")

#     json_midi = json.load(midi)

#     return(json_midi)