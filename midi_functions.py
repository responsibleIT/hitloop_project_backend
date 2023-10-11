# File by Erik Slingerland
# Contains functions used for transforming MIDI files to a format usefull for Machine Learning and interpretable by tone.js

# Parts are:
# 1. Midi to Array. This is for AI representation
# 2. Array to Json. This is for changing the Array in a readable format for tone.js


####Packages####
import numpy as np
import os
import string
from pathlib import Path
from matplotlib import pyplot as plt
import mido
import json
import random
from scipy import stats

target_ticks_per_beat = None
current_ticks_per_beat = None
include_rhythm = None
include_bass = None
include_accompany = None
include_melody = None
continuous_velocity = None


######################################################################################
def rhythm_dict():
    """Function for consitently giving the same lookup table for drums. Each entry is the corresponding MIDI channel for each entry in the drum area
    """
    # drum_lookup_list = [36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 48, 50, 51, 53, 54, 56, 60, 61, 62, 63, 69, 70, 76, 80, 81, 82]
    rhythm_lookup_list = [42, 35, 38]
    # rhythm_lookup_list = list(range(21,109))
    return rhythm_lookup_list


def bass_dict():
    """Function for consitently giving the same lookup table for rhythm. Each entry is the corresponding MIDI channel for each entry in the rhythm area
    """
    bass_lookup_list = [64, 35, 38, 43, 53, 52, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    # bass_lookup_list = list(range(21,109))
    return bass_lookup_list


def accompany_dict():
    """Function for consitently giving the same lookup table for lead. Each entry is the corresponding MIDI channel for each entry in the lead area
    """
    accompany_lookup_list = [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                             45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
                             68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                             91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 107]
    # accompany_lookup_list = list(range(21,109))
    return accompany_lookup_list


def melody_dict():
    """Function for consitently giving the same lookup table for lead. Each entry is the corresponding MIDI channel for each entry in the lead area
    """
    melody_lookup_list = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                          44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
                          67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                          90, 91, 92, 93, 94, 95, 96, 97, 98, 100, 101, 102]
    # melody_lookup_list = list(range(21,109))
    return melody_lookup_list


#############################################################################################
##########################Part 1. Midi to Array##############################################

############################
#######sub_functions########
############################
# Static functions
# Code adapted from https://medium.com/analytics-vidhya/convert-midi-file-to-numpy-array-in-python-7d00531890c
# Converts MIDI message to dictionary of signal, time and note
def msg2dict(msg):
    result = dict()
    if 'note_on' in msg:
        on_ = True
    elif 'note_off' in msg:
        on_ = False
    else:
        on_ = None
    result['time'] = int(msg[msg.rfind('time'):].split(' ')[0].split('=')[1].translate(
        str.maketrans({a: None for a in string.punctuation})))

    if on_ is not None:
        for k in ['note', 'velocity']:
            result[k] = int(msg[msg.rfind(k):].split(' ')[0].split('=')[1].translate(
                str.maketrans({a: None for a in string.punctuation})))
    return [result, on_]


# Creates a switch for when to turn on/off a note
def switch_note(last_state, note, velocity, on_=True):
    # If last state was empty, create empty row
    result = [0] * 128 if last_state is None else last_state.copy()
    result[note] = velocity if on_ else 0
    return result


# Combines dictionary and switch note
def get_new_state(new_msg, last_state):
    new_msg, on_ = msg2dict(str(new_msg))
    new_state = switch_note(last_state, note=new_msg['note'], velocity=new_msg['velocity'],
                            on_=on_) if on_ is not None else last_state
    return [new_state, new_msg['time']]


# Creates sequence of new states across time and creates empty rows
def track2seq(track):
    result = []
    last_state, last_time = get_new_state(str(track[0]), [0] * 128)
    for i in range(1, len(track)):
        new_state, new_time = get_new_state(track[i], last_state)
        if new_time > 0:
            result += [last_state] * new_time
        last_state, last_time = new_state, new_time
    return result


# Converts full MIDI into Raw_Array
def mid2arry(mid, min_msg_pct=0.1):
    tracks_len = [len(tr) for tr in mid.tracks]
    min_n_msg = max(tracks_len) * min_msg_pct
    # convert each track to nested list
    all_arys = []
    for i in range(len(mid.tracks)):
        if len(mid.tracks[i]) > min_n_msg:
            ary_i = track2seq(mid.tracks[i])
            all_arys.append(ary_i)
    # make all nested list the same length
    max_len = max([len(ary) for ary in all_arys])
    for i in range(len(all_arys)):
        if len(all_arys[i]) < max_len:
            all_arys[i] += [[0] * 128] * (max_len - len(all_arys[i]))
    all_arys = np.array(all_arys)
    all_arys = all_arys.max(axis=0)
    # trim: remove consecutive 0s in the beginning and at the end
    sums = all_arys.sum(axis=1)
    ends = np.where(sums > 0)[0]
    return all_arys[min(ends): max(ends)]


# Converts only track 9 and 10 of MIDI into Raw_Array. This is for the rhythm file to ensure only drum track
def mid2arry_drums(mid, min_msg_pct=0.1):
    tracks_len = [len(tr) for tr in mid.tracks]
    min_n_msg = max(tracks_len) * min_msg_pct
    # convert each track to nested list
    all_arys = []
    for i in range(len(mid.tracks)):
        if len(mid.tracks[i]) > min_n_msg:
            # Filter to only channel 9 and 10
            if mid.tracks[i][1].channel == 9 or mid.tracks[i][1].channel == 10:
                ary_i = track2seq(mid.tracks[i])
                all_arys.append(ary_i)
    # make all nested list the same length
    max_len = max([len(ary) for ary in all_arys])
    for i in range(len(all_arys)):
        if len(all_arys[i]) < max_len:
            all_arys[i] += [[0] * 128] * (max_len - len(all_arys[i]))
    all_arys = np.array(all_arys)
    all_arys = all_arys.max(axis=0)
    # trim: remove consecutive 0s in the beginning and at the end
    sums = all_arys.sum(axis=1)
    ends = np.where(sums > 0)[0]
    return all_arys[min(ends): max(ends)]


def convert_midisong_2array(folder_location: str, target_length: int = 2048):
    """Converts a folder containing: lead, drums and rhythm midi into 1 array. Currently dimension of (target_length,381)

    Args:
        folder_location (str): _description_
        target_length (int, optional): _description_. Defaults to 2048.
    """

    midi_drum, midi_rhythm, midi_lead = get_midi_from_folder(folder_location)

    drum_lookup = drum_dict()
    rhythm_lookup = rhythm_dict()
    lead_lookup = lead_dict()

    if type(midi_drum) == mido.midifiles.midifiles.MidiFile:
        midi_drum = clean_array(mid2arry(midi_drum), midi_drum.ticks_per_beat, target_length)
        midi_drum = midi_drum[:, drum_lookup]

    if type(midi_rhythm) == mido.midifiles.midifiles.MidiFile:
        midi_rhythm = clean_array(mid2arry(midi_rhythm), midi_rhythm.ticks_per_beat, target_length)
        midi_rhythm = midi_rhythm[:, rhythm_lookup]
    if type(midi_lead) == mido.midifiles.midifiles.MidiFile:
        midi_lead = clean_array(mid2arry(midi_lead), midi_lead.ticks_per_beat, target_length)
        midi_lead = midi_lead[:, lead_lookup]

    complete_array = combine_arrays(midi_drum, midi_rhythm, midi_drum)

    return complete_array


def get_midi_from_folder(folder_location: str):
    """Function for loading the midi_drum, midi_rhythm and midi_lead data from a specific folder
    If one of the files doesn't exist. An empty array of 2048,128 is returned

    Args:
        folder_location (str): Folder location 
    """

    target_x = target_ticks_per_beat * 4
    try:
        midi_rhythm = mido.MidiFile(os.path.join(folder_location, 'drums.mid'))

    except:
        print('drums.mid was not found, creating empty array')
        midi_rhythm = np.empty((target_x, 128))

    try:
        midi_bass = mido.MidiFile(os.path.join(folder_location, 'bass.mid'))

    except:
        print('bass.mid was not found, creating empty array')
        midi_bass = np.empty((target_x, 128))

    try:
        midi_accompany = mido.MidiFile(os.path.join(folder_location, 'accompany.mid'))

    except:
        print('accompany.mid was not found, creating empty array')
        midi_accompany = np.empty((target_x, 128))

    try:
        midi_melody = mido.MidiFile(os.path.join(folder_location, 'melody.mid'))

    except:
        print('melody.mid was not found, creating empty array')
        midi_melody = np.empty((target_x, 128))

    return midi_rhythm, midi_bass, midi_accompany, midi_melody


def return_bars(input_array: np.array):
    """
    Function converts input_tickrate to target_tickrate
    normalize the array's velocity.
    And return the full bars we have

    Returns a list of arrays containing all full bars in a song. Size of each bar is target_ticks_per_beat*4,128

    Args:
        input_array (np.array): Input array to be converted into standardised output
    """

    current_length = np.shape(input_array)[0]
    n_columns = np.shape(input_array)[1]

    target_length = int(np.floor(current_length / current_ticks_per_beat * target_ticks_per_beat))

    new_array = np.empty((target_length, n_columns))

    # Change length of each column(note) to the target_length
    for column in range(0, n_columns):
        audio_row = input_array[:, column]
        # Decide how often to devide
        n = np.floor_divide(current_length, target_length)
        if n > 0:
            new_audio_row = audio_row[:n * target_length].reshape((n, target_length)).mean(axis=0)
            # Extra ensurance of output size
            new_audio_row = new_audio_row[:target_length]

            new_array[:, column] = new_audio_row
        else:
            new_audio_row = audio_row[:target_length]
            new_array[:, column] = new_audio_row

    # Return bars
    bar_length = 4 * target_ticks_per_beat
    bars = []
    # select every 200 rows until you can no longer get any full length rows
    i = 0
    while i + bar_length <= new_array.shape[0]:
        selected_rows = new_array[i:i + bar_length]
        bars.append(selected_rows)
        # do something with the selected_rows
        i += bar_length

    return bars


def convert_midi_2array(folder_location: str):
    """Converts a folder containing: lead, drums and rhythm midi into 1 array. Currently dimension of (target_length,381)

    Args:
        folder_location (str): location of MIDI files
    """

    midi_rhythm, midi_bass, midi_accompany, midi_melody = get_midi_from_folder(folder_location)

    midi_rhythm = mid2arry_drums(midi_rhythm)
    if continuous_velocity:
        normalization_factor = 1 / 128
        midi_rhythm = midi_rhythm * normalization_factor

    else:
        midi_rhythm[midi_rhythm != 0] = 1

    midi_bass = mid2arry(midi_bass)
    if continuous_velocity:
        normalization_factor = 1 / 128
        midi_bass = midi_bass * normalization_factor

    else:
        midi_bass[midi_bass != 0] = 1

    midi_accompany = mid2arry(midi_accompany)
    if continuous_velocity:
        normalization_factor = 1 / 128
        midi_accompany = midi_accompany * normalization_factor

    else:
        midi_accompany[midi_accompany != 0] = 1

    midi_melody = mid2arry(midi_melody)
    if continuous_velocity:
        normalization_factor = 1 / 128
        midi_melody = midi_melody * normalization_factor

    else:
        midi_melody[midi_melody != 0] = 1

    return midi_rhythm, midi_bass, midi_accompany, midi_melody


def combine_array(midi_rhythm: np.array, midi_bass: np.array, midi_accompany: np.array,
                  midi_melody: np.array) -> np.array:
    """Function for combining the different arrays
    Input are rhythm, bass, accompany and melody

    Args:
        midi_rhythm (np.array): Arr
        midi_bass (np.array): _description_
        midi_accompany (np.array): _description_
        midi_melody (np.array): _description_

    Returns:
        np.array: Output array where X is the longest input array length and y is the combination of rhythm_dict, bass_dict, accompany_dict and melody_dict. Order of array is: rhythm, bass, accompany and melody
    """
    # To subscript channels
    midi_rhythm_small = midi_rhythm[:, rhythm_dict()]
    midi_bass_small = midi_bass[:, bass_dict()]
    midi_accompany_small = midi_accompany[:, accompany_dict()]
    midi_melody_small = midi_melody[:, melody_dict()]

    shape_rhythm = np.shape(midi_rhythm_small)
    shape_bass = np.shape(midi_bass_small)
    shape_accompany = np.shape(midi_accompany_small)
    shape_melody = np.shape(midi_melody_small)

    # deciding demensions of output
    x_length = max(shape_rhythm[0], shape_bass[0], shape_accompany[0], shape_melody[0])
    y_length = 0

    if include_rhythm:
        y_length += shape_rhythm[1]

    if include_bass:
        y_length += shape_bass[1]

    if include_accompany:
        y_length += shape_accompany[1]

    if include_melody:
        y_length += shape_melody[1]

    combined_array = np.empty((x_length, y_length))

    # Add the info from the midi files
    i = 0

    if include_rhythm:
        for column in midi_rhythm_small.T:
            combined_array[:shape_rhythm[0], i] = column
            i = i + 1

    if include_bass:
        for column in midi_bass_small.T:
            combined_array[:shape_bass[0], i] = column
            i = i + 1

    if include_accompany:
        for column in midi_accompany_small.T:
            combined_array[:shape_accompany[0], i] = column
            i = i + 1

    if include_melody:
        for column in midi_melody_small.T:
            combined_array[:shape_melody[0], i] = column
            i = i + 1

    print('First ' + str(i) + ' columns of combined_array filled with data')
    return combined_array


#############################
#######Main_functions########
#############################


def folder_to_data_pipeline(song_path: str, target_ticks_per_beat_var: int = 32, current_ticks_per_beat_var: int = 200,
                            output_bars_var: bool = True, include_rhythm_var: bool = True,
                            include_bass_var: bool = True, include_accompany_var: bool = True,
                            include_melody_var: bool = True, continuous_velocity_var: bool = True):
    """General function from walking the pipeline of a folder with midi_files to a final output file

    Args:
        song_path (str): Path to folder which contains drums.mid, bass.mid, accompany.mid and melody.mid of 1 song
        target_ticks_per_beat (int, optional): Decides the ticks_per_beat of the resulting array(s). Defaults to 32.
        current_ticks_per_beat (int, optional): The ticks_per_beat that the input files adhere to. Defaults to 200.
        output_bars (bool, optional): Decides whether the output of the function is the bars in a list or just the full array. Defaults to True.
        include_rhythm (bool, optional): Decides if the rhythm midi is included in the final array. Defaults to True.
        include_bass (bool, optional): Decides if the bass midi is included in the final array. Defaults to True.
        include_accompany (bool, optional): Decides if the accompany midi is included in the final array. Defaults to True.
        include_melody (bool, optional): Decides if the melody midi is included in the final array. Defaults to True.
        continuous_velocity (bool, optional): Decides the velocity scale. If True the velocity is a continuous scale between 0 and 1 (0 is off, 1 is full velocity). If False, velocity is a binary choice of 0 and 1. Defaults to True.

    Returns:
     if output_bars = True: List of all bars in the input song
     if output_bars = False: Np.array with all channels of the input song
    """

    # Ensure global variable status for important info

    global target_ticks_per_beat
    target_ticks_per_beat = target_ticks_per_beat_var
    global current_ticks_per_beat
    current_ticks_per_beat = current_ticks_per_beat_var
    global include_rhythm
    include_rhythm = include_rhythm_var
    global include_bass
    include_bass = include_bass_var
    global include_accompany
    include_accompany = include_accompany_var
    global include_melody
    include_melody = include_melody_var
    global continuous_velocity
    continuous_velocity = continuous_velocity_var
    # Pipeline
    midi_rhythm, midi_bass, midi_accompany, midi_melody = convert_midi_2array(song_path)
    output = combine_array(midi_rhythm, midi_bass, midi_accompany, midi_melody)

    if output_bars_var:
        output = return_bars(output)

    return output


#############################################################################################
#############################################################################################
##########################Part 2. Array to Json##############################################

############################
#######sub_functions########
############################

# Function to do this

# Function to do this
def resize_ouput_length(input_array: np.array, output_length: int = 16):
    """Function for adjusting the output length of the array oto a format appropriate for the sequencer

    Args:
        input_array (np.array): Array to be reshaped
        output_length (int, optional): The length of the output array. make sure it is the same as the sequencer length. Defaults to 16.

    Returns:
        np.array: The new array with dimensions [output_lenght,original_n_columns]
    """

    array_shape = np.shape(input_array)

    new_array = np.zeros((output_length, array_shape[1]))

    group_size = array_shape[0] // output_length

    for i in range(array_shape[1]):
        temp_row = input_array[:, i]
        groups = [temp_row[i:i + group_size] for i in range(0, len(temp_row), group_size)]
        new_row = np.array([int(stats.mode(group)[0][0]) for group in groups])
        new_array[:, i] = new_row

    return new_array


def compress_bass(input_array: np.array):
    """Function to compress the output of the bass to 1 row. This will always be the one on index 0 of bass_lookup for now.
    This is not used in the process of transforming midi to json, it would be a pre-processing step

    Args:
        input_array (np.array): Output of the model

    Returns:
        np.array: Array where the bass channel is empty except for index 0. This is filled with the rhythm of the bass
    """
    rhythm_lookup = rhythm_dict()
    bass_lookup = bass_dict()
    bass_notes = input_array[:, len(rhythm_lookup):(len(rhythm_lookup) + len(bass_lookup))]

    # Create temp arrays
    bass_shape = np.shape(bass_notes)
    bass_output = np.zeros(shape=bass_shape)

    compressed_array = np.zeros(shape=bass_shape[0])

    for row in range(bass_shape[0]):
        all_cols = bass_notes[row, :]
        max_velocity = np.max(all_cols)
        if max_velocity > 0:
            compressed_array[row] = max_velocity

    # Add to note 0 of bass
    bass_output[:, 0] = compressed_array

    output_array = input_array
    output_array[:, len(rhythm_lookup):(len(rhythm_lookup) + len(bass_lookup))] = bass_output

    return output_array


def midi_lookup_table() -> list:
    """function for a table to look up note names and MIDI number codes.

    Returns:
        list: Returns MIDI lookup table result
    """
    # Create Name - midicode lookup table 12 notes per octave. for -1 - 9 with all of them. For simplicity we will only use sharps, not flats
    # We will Do this for C#-1 - C9. The others will be added manually since it is more trouble to do that automatically

    pattern = ['C',
               'C#',
               'D',
               'D#',
               'E',
               'F',
               'F#',
               'G',
               'G#',
               'A',
               'A#',
               'B', ]

    octaves = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    lookup_table = []
    for i in octaves:
        for note in pattern:
            note_to_add = f'{note}{i}'
            lookup_table.append(note_to_add)

    final_list = ['C9', 'C#9', 'D9', 'D#9', 'E9', 'F9', 'F#9', 'G9']
    for note in final_list:
        lookup_table.append(note)

    return lookup_table


def extract_events(input_array: np.array, tick_per_beat: int, beats_per_minute: int) -> list:
    """Function that creates a list of dictionaries based on the input_array. These dictionaries are modeled after the required information for a json input in tone.js

    Args:
        input_array (np.array): Input track to be extracted
        tick_per_beat (int): Track per beat input time
        beats_er_minute (int): BPM for the output file

    Returns:
        list: Returns a list of dictionaries where each dictionary is a note played in the input array.
    """
    events = []
    size_track = np.shape(input_array)

    time_per_tick = 60 / beats_per_minute / tick_per_beat

    MIDI_nr_names = midi_lookup_table()

    # Checks for each tick if there are events
    for tick in range(size_track[0]):
        tick_events = input_array[tick]
        value_indexes = np.nonzero(tick_events)[0]
        # If there is an event, for each note where there is an event.
        if len(value_indexes) != 0:
            for note in value_indexes:
                note_array = input_array[:, note]

                # If the note is on tick 0 or if the previous note is 0 to ensure we don't create an event for each tick of a note.
                current_velocity = note_array[(tick)]
                previous_velocity = note_array[(tick - 1)]
                if tick == 0 or current_velocity != previous_velocity:

                    # Decide time and ticks field
                    start_tick = tick
                    if tick == 0:
                        start_time = 0.0
                    else:
                        start_time = start_tick * time_per_tick

                    # Decide midi and name field
                    midi_nr = note
                    name_field = MIDI_nr_names[midi_nr]

                    # Decide duration and duration ticks field
                    duration_ticks = tick + np.argmax(note_array[tick:] == 0)
                    duration = duration_ticks * time_per_tick

                    # Velocity for now is the velocity of the first tick
                    velocity = note_array[tick]
                    if velocity > 0.95:
                        velocity = 1.0
                    if velocity < 0.001:
                        velocity = 0

                    event_dictionary = {
                        "duration": float(duration),
                        "durationTicks": int(duration_ticks),
                        "midi": int(midi_nr),
                        "name": str(name_field),
                        "ticks": int(start_tick),
                        "time": float(start_time),
                        "velocity": float(velocity)
                    }
                    # Min length of note is 10 ticks here. AKA roudnly 0.04 seconds
                    if duration_ticks > 10 and velocity > 0.02:
                        events.append(event_dictionary)

    return events


def create_track(input_array: np.array, input_track_events: list, instrument: 'str' = 'drums',
                 channel: int = 10) -> dict:
    """Function for creating a track information dictionary for a single instrument

  Args:
      input_track_events (np.array): Array of a single instrument same as sent to extract_events
      input_track_events (list): List of dictionaries of a single instrument as given by extract_events()
      instrument (str, optional): Decides the instrument name of the midi tack. Defaults to 'drums' for debugging
      channel (int, optional): Decides the output MIDI channel of this track. Defaults to 10 for debugging

  Returns:
      dict: Dictionary that contains the information for a single track
  """
    instrument_track = {'channel': channel,
                        "controlChanges": {},
                        "pitchBends": [],
                        "instrument": {
                            "family": instrument,
                            "number": 0,
                            "name": "standard kit"
                        },
                        "name": str(instrument + '_output'),
                        'notes': input_track_events,
                        'endOfTrackTicks': np.shape(input_array)[0]
                        }
    return instrument_track


#############################
#######Main_functions########
#############################

def convert_array2json(input_array: np.array, output_beats_per_minute: int = 120, input_tick_per_beat: int = 32,
                       midi_name: str = 'placeholder') -> str:
    """Function for transforming an array with 1-4 channels (rhythm, bass, accompany, melody) into a JSON format that is readable by tone.js


    Args:
        input_array (np.array): Input array to transfrom from the model to json
        output_beats_per_minute (int, optional): Beats per minute of output file. Defaults to 120.
        input_tick_per_beat (int, optional): Tick per beat used by the input array. Used to convert into time. Defaults to 32.
        midi_name (str, optional): Name of the resulting midi file in the header. Defaults to 'placeholder'.

    Returns:
        str: _description_
    """

    ####Header information####
    header = {
        "keySignatures": [],
        "meta": [],
        "name": midi_name,
        "ppq": input_tick_per_beat,
        "tempos": [],
        "timeSignatures": [
            {
                "ticks": 0,
                "timeSignature": [
                    4,
                    4
                ],
                "measures": 0
            }
        ]
    }

    ####Track information####
    # 1 bar version
    track_length = input_tick_per_beat * 4

    # Seperate tracks

    rhythm_lookup = rhythm_dict()
    bass_lookup = bass_dict()
    accompany_lookup = accompany_dict()
    melody_lookup = melody_dict()

    input_shape = np.shape(input_array)

    input_channels = []

    if input_shape[1] == (len(rhythm_lookup)):
        print('only_rhythm')
        rhythm_input = input_array[:, 0:len(rhythm_lookup)]
        input_channels.append(rhythm_input)

    elif input_shape[1] == (len(rhythm_lookup) + len(bass_lookup)):
        print('rhythm and bass')
        rhythm_input = input_array[:, 0:len(rhythm_lookup)]
        bass_input = input_array[:, len(rhythm_lookup):(len(rhythm_lookup) + len(bass_lookup))]

        input_channels.append(rhythm_input)
        input_channels.append(bass_input)

    elif input_shape[1] == (len(rhythm_lookup) + len(bass_lookup) + len(accompany_lookup)):
        print('rhythm, bass and accompany')
        rhythm_input = input_array[:, 0:len(rhythm_lookup)]
        bass_input = input_array[:, len(rhythm_lookup):(len(rhythm_lookup) + len(bass_lookup))]
        accompany_input = input_array[:, (len(rhythm_lookup) + len(bass_lookup)):(
                    len(rhythm_lookup) + len(bass_lookup) + len(accompany_lookup))]

        input_channels.append(rhythm_input)
        input_channels.append(bass_input)
        input_channels.append(accompany_input)


    elif input_shape[1] == (len(rhythm_lookup) + len(bass_lookup) + len(accompany_lookup) + len(melody_lookup)):
        print('rhythm, bass accompany and melody')
        rhythm_input = input_array[:, 0:len(rhythm_lookup)]
        bass_input = input_array[:, len(rhythm_lookup):(len(rhythm_lookup) + len(bass_lookup))]
        accompany_input = input_array[:, (len(rhythm_lookup) + len(bass_lookup)):(
                    len(rhythm_lookup) + len(bass_lookup) + len(accompany_lookup))]
        melody_input = input_array[:, (len(rhythm_lookup) + len(bass_lookup) + len(accompany_lookup)):(
                    len(rhythm_lookup) + len(bass_lookup) + len(accompany_lookup) + len(melody_lookup))]

        input_channels.append(rhythm_input)
        input_channels.append(bass_input)
        input_channels.append(accompany_input)
        input_channels.append(melody_input)
    else:
        print('input shape not applicable to this command. No json was made')
        return

    rhythm_new = np.zeros_like(np.empty((track_length, 128)))
    bass_new = np.zeros_like(np.empty((track_length, 128)))
    accompany_new = np.zeros_like(np.empty((track_length, 128)))
    melody_new = np.zeros_like(np.empty((track_length, 128)))

    # if rhythm channel exists
    for i in range(0, len(input_channels)):
        # Rhythm channel
        if i == 1:
            for i in range(0, len(rhythm_lookup)):
                column = rhythm_input[:, i]
                midi_code = rhythm_lookup[i]
                rhythm_new[:, midi_code] = column

        if i == 2:
            for i in range(0, len(bass_lookup)):
                column = bass_input[:, i]
                midi_code = bass_lookup[i]
                bass_new[:, midi_code] = column

        if i == 3:
            for i in range(0, len(accompany_lookup)):
                column = accompany_input[:, i]
                midi_code = accompany_lookup[i]
                accompany_new[:, midi_code] = column

        if i == 4:
            for i in range(0, len(melody_lookup)):
                column = melody_input[:, i]
                midi_code = melody_lookup[i]
                melody_new[:, midi_code] = column

    list_of_tracks = [rhythm_new, bass_new, accompany_new, melody_new]
    track_names = [['drums', 10], ['bass', 1], ['accompany', 2], ['melody', 3]]

    # create events list for each instrument
    events = []
    for track in range(0, len(list_of_tracks)):
        temp_events = extract_events(list_of_tracks[track], tick_per_beat=input_tick_per_beat,
                                     beats_per_minute=output_beats_per_minute)
        events.append(temp_events)

    tracks = []

    for track in range(0, len(events)):
        if len(events[track]) >= 1:
            temp_track = create_track(input_array=list_of_tracks[track], input_track_events=events[track],
                                      instrument=track_names[track][0], channel=track_names[track][1])
            tracks.append(temp_track)

    output_dictionary = {"header": header,
                         "tracks": tracks}

    output_json = json.dumps(output_dictionary)

    return output_json


#############################################################################################
#############################################################################################
##########################Part 3. Midi evaluation functions##################################

def calculate_unqualified_notes(input_array: np.array) -> int:
    """Function to calculate amount of unqualified notes. This is defined from the MuseGAN paper where a note shorter then a 32nd note is unqualified.

    Args:
        input_array (np.array): Input array for which the amount of unqualified notes are counted. Array has to be 1 bar of 4/4 beats long to ensure correct counting of note time.

    Returns:
        int: Amount of unqualified notes
    """

    shape = np.shape(input_array)
    input_ticks_per_beat = int(shape[0] / 4)
    qualified_note = input_ticks_per_beat / 8

    unqualified = []

    for i in range(shape[1]):
        row = input_array[:, i]
        sub_arrays = np.split(row, np.where(row == 0)[0])
        note_lengths = [np.count_nonzero(sub_array) for sub_array in sub_arrays if np.count_nonzero(sub_array) > 0]
        unqualified_count = len([x for x in note_lengths if x < qualified_note])
        unqualified.append(unqualified_count)

    n_unqualified = sum(unqualified)
    return n_unqualified


def calculate_empty_ratio(input_array: np.array) -> float:
    """Function to calculate ratio of generated MIDI file that has no note playing

    Args:
        input_array (np.array): input array where it should be calculated from
    Returns:
        float: ratio of how much is empty. 1 is an empty array and 0 is a fully filled array
    """

    shape = np.shape(input_array)

    n_empty_rows = 0

    for i in range(shape[0]):
        time_point = input_array[i, :]
        if np.all(time_point == 0):
            n_empty_rows += 1

    ratio = n_empty_rows / shape[0]
    return ratio

#############################################################################################
#############################################################################################
