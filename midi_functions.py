#File by Erik Slingerland
#Contains functions used for transforming MIDI files to a format usefull for Machine Learning and interpretable by tone.js

#Parts are: 
#1. Midi to Array. This is for AI representation
#2. Array to Json. This is for changing the Array in a readable format for tone.js


####Packages####
import numpy as np
import os
import string 
from pathlib import Path
from matplotlib import pyplot as plt
import mido
import json


######################################################################################
def drum_dict():
    """Function for consitently giving the same lookup table for drums. Each entry is the corresponding MIDI channel for each entry in the drum area
    """    
    drum_lookup_list = [36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 48, 50, 51, 53, 54, 56, 60, 61, 62, 63, 69, 70, 76, 80, 81, 82]
    return drum_lookup_list

def rhythm_dict():
    """Function for consitently giving the same lookup table for rhythm. Each entry is the corresponding MIDI channel for each entry in the rhythm area
    """    
    rhythm_lookup_list = list(range(0,128))
    return rhythm_lookup_list

def lead_dict():
    """Function for consitently giving the same lookup table for lead. Each entry is the corresponding MIDI channel for each entry in the lead area
    """    
    lead_lookup_list = list(range(0,128))
    return lead_lookup_list

#############################################################################################
##########################Part 1. Midi to Array##############################################

############################
#######sub_functions########
############################
def get_midi_from_folder(folder_location:str):
    """Function for loading the midi_drum, midi_rhythm and midi_lead data from a specific folder
    If one of the files doesn't exist. An empty array of 2048,128 is returned

    Args:
        folder_location (str): Folder location 
    """ 
    try:
        midi_drum = mido.MidiFile(str(folder_location + '/' + 'drums.mid'))
    
    except:
        print('drums.mid was not found, creating empty array')
        midi_drum = np.empty((2048,128))
    
    try:
        midi_rhythm = mido.MidiFile(str(folder_location + '/'  + 'rhythm.mid'))
    
    except:
        print('rhythm.mid was not found, creating empty array')
        midi_rhythm = np.empty((2048,128))
    
    try:
        midi_lead = mido.MidiFile(str(folder_location + '/'  + 'lead.mid'))
    
    except:
        print('lead.mid was not found, creating empty array')
        midi_lead = np.empty((2048,128))
    
    return midi_drum, midi_rhythm, midi_lead


# Code adapted from https://medium.com/analytics-vidhya/convert-midi-file-to-numpy-array-in-python-7d00531890c
#Converts MIDI message to dictionary of signal, time and note
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
    new_state = switch_note(last_state, note=new_msg['note'], velocity=new_msg['velocity'], on_=on_) if on_ is not None else last_state
    return [new_state, new_msg['time']]

# Creates sequence of new states across time and creates empty rows
def track2seq(track):
    result = []
    last_state, last_time = get_new_state(str(track[0]), [0]*128)
    for i in range(1, len(track)):
        new_state, new_time = get_new_state(track[i], last_state)
        if new_time > 0:
            result += [last_state]*new_time
        last_state, last_time = new_state, new_time
    return result

#Converts full MIDI into Raw_Array
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


def clean_array(input_array:np.array, ticks_per_beat:int, target_length:int = 2048):
    """Function to normalize the array's from the midi-array pipeline.
    Target size is 4 bars. If too few bars are entered, we will add them together, if too many are used, we will create a cut-off at the end of the 4th bar

    Returns new array of 2048,y which is a 4 bars of 4/4 time

    Args:
        input_array (np.array): Input array to be converted into standardised output
        ticks_per_beat (np.array): Ticks per beat of input array. Taken from original midi file
        target_length (int, optional): Defines the output length for each audio channel. Defaults to 2048. This is 128 elements for each beat
    """    
    new_array = np.empty((2048,128))
    current_length = np.shape(input_array)[0]

    #Current no of bars, Assuming all bars are 4/4
    current_bars = int(np.ceil(current_length/ticks_per_beat/4))

    #For too few bars
    original_array = input_array
    if current_bars < 4:
        grow_factor = int(4/current_bars-1)
        for i in range(0,grow_factor):
            input_array = np.concatenate((input_array, original_array), axis=0)

    #For too many bars
    elif current_bars > 4:
        four_bar_length = ticks_per_beat*4*4
        input_array = original_array[0:four_bar_length,:]


    #Change length of each column(note) to the target_length
    n_columns = np.shape(input_array)[1]
    for column in range(0,n_columns):
        audio_row = input_array[:,column]
        #Decide how often to devide
        n = np.floor_divide(current_length,target_length)
        if n >0:
            new_audio_row = audio_row[:n * target_length].reshape((n, target_length)).mean(axis=0)
            #Extra ensurance of output size
            new_audio_row = new_audio_row[:target_length]

            new_array[:,column] = new_audio_row
        else:
            new_audio_row = audio_row[:target_length]
            new_array[:,column] = new_audio_row
    

    #Normalize Velocity with normalization factor
    normalization_factor = 1/128
    new_array = new_array * normalization_factor

    return(new_array)



def combine_arrays(drum_array:np.array, rhythm_array:np.array, lead_array:np.array):
    """Inputs the 3 instruments with normalised array's
    Output is 1 array where channel at index 0-25 is drums, 128-253 is rhythm and 254-380 is rhythm.

    Args:
        drum_array (np.array): Array of the drum section
        rhythm_array (np.array): Array of the rhythm section (placeholder)
        lead_array (np.array): Array of the lead section (placeholder)
    """    
    final_array = np.concatenate((drum_array, rhythm_array, lead_array), axis=1)

    return final_array

#############################
#######Main_functions########
#############################
def convert_midisong_2array(folder_location:str, target_length:int = 2048):
    """Converts a folder containing: lead, drums and rhythm midi into 1 array. Currently dimension of (target_length,381)

    Args:
        folder_location (str): _description_
        target_length (int, optional): _description_. Defaults to 2048.
    """  
      
    midi_drum, midi_rhythm, midi_lead =get_midi_from_folder(folder_location)

    drum_lookup = drum_dict()
    rhythm_lookup = rhythm_dict()
    lead_lookup = lead_dict()

    if type(midi_drum) == mido.midifiles.midifiles.MidiFile:
        midi_drum = clean_array(mid2arry(midi_drum), midi_drum.ticks_per_beat, target_length)
        midi_drum = midi_drum[:,drum_lookup]
    
    if type(midi_rhythm) == mido.midifiles.midifiles.MidiFile:
        midi_rhythm = clean_array(mid2arry(midi_rhythm), midi_rhythm.ticks_per_beat, target_length)
        midi_rhythm = midi_rhythm[:,rhythm_lookup]
    if type(midi_lead) == mido.midifiles.midifiles.MidiFile:
        midi_lead = clean_array(mid2arry(midi_lead), midi_lead.ticks_per_beat, target_length)
        midi_lead = midi_lead[:,lead_lookup]
    
    complete_array = combine_arrays(midi_drum, midi_rhythm, midi_drum)

    return complete_array


#############################################################################################
#############################################################################################
##########################Part 2. Array to Json##############################################

############################
#######sub_functions########
############################
def midi_lookup_table()-> list:
    """function for a table to look up note names and MIDI number codes.

    Returns:
        list: Returns MIDI lookup table result
    """    
    # Create Name - midicode lookup table 12 notes per octave. for -1 - 9 with all of them. For simplicity we will only use sharps, not flats
    # We will Do this for C#-1 - C9. The others will be added manually since it is more trouble to do that automatically
    start_notes = ['C#-1', 'D-1', 'D#-1', 'E-1', 'F-1', 'F#-1', 'G-1', 'G#-1', 'A-1', 'A#-1','B-1']
    end_notes = ['C9', 'C#9', 'D9', 'D#9', 'E9', 'F9', 'F#9', 'G9', 'G#9']
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octaves = [0,1,2,3,4,5,6,7,8]

    MIDI_nr_names = start_notes
    for number in octaves:
        temp = []
        for note in notes:
            new_note = str(note + str(number))
            MIDI_nr_names.append(new_note)
    for note in end_notes:

        MIDI_nr_names.append(note)
    return MIDI_nr_names



def extract_events(input_array:np.array, beats_per_minute:int = 120, tick_per_beat:int = 128)-> list:
    """Function that creates a list of dictionaries based on the input_array. These dictionaries are modeled after the required information for a json input in tone.js

    Args:
        input_array (np.array): Input track to be extracted
        beats_per_minute (int, optional): BPM of the output song. Defaults to 120.
        tick_per_beat (int, optional): Ticks per beat says how many ticks(numbers) there are for every beat. Defaults to 128 because this is the same as the current model.

    Returns:
        list: Returns a list of dictionaries where each dictionary is a note played in the input array.
    """    
    events = []
    size_track = np.shape(input_array)

    time_per_tick = 60/beats_per_minute/tick_per_beat

    MIDI_nr_names = midi_lookup_table()

    #Checks for each tick if there are events
    for tick in range(size_track[0]):
        tick_events = input_array[tick]
        value_indexes = np.nonzero(tick_events)[0]
        #If there is an event, for each note where there is an event.
        if len(value_indexes) != 0:
            for note in value_indexes:
                note_array = input_array[:, note]

                #If the note is on tick 0 or if the previous note is 0 to ensure we don't create an event for each tick of a note.
                current_velocity = note_array[(tick)]
                previous_velocity = note_array[(tick-1)]
                if tick == 0 or current_velocity != previous_velocity:

                    #Decide time and ticks field
                    start_tick = tick
                    if tick == 0:
                        start_time = 0.0
                    else:
                        start_time = start_tick*time_per_tick

                    #Decide midi and name field
                    midi_nr = note
                    name_field = MIDI_nr_names[midi_nr]

                    #Decide duration and duration ticks field
                    duration_ticks = tick + np.argmax(note_array[tick:] == 0)
                    duration = duration_ticks*time_per_tick

                    #Velocity for now is the velocity of the first tick
                    velocity = note_array[tick]

                    event_dictionary = {
                "duration": float(duration),
                "durationTicks": int(duration_ticks),
                "midi": int(midi_nr),
                "name": str(name_field),
                "ticks": int(start_tick),
                "time": float(start_time),
                "velocity": float(velocity)
            }
                    #Min length of note is 10 ticks here. AKA roudnly 0.04 seconds
                    if duration_ticks > 10:
                        events.append(event_dictionary)
            
    return events

def create_track(input_array:np.array, input_track_events:list, instrument:'str' = 'drums', channel:int = 10)->dict:
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

def convert_array2json(input_array:np.array, beats_per_minute:int = 120, tick_per_beat:int = 4, midi_name:str = 'placeholder') -> str:
    """Function for transforming an array with 3 channels (drums,rhythm,lead) into a JSON format that is readable by tone.js

    Args:
        input_array (np.array): Array to be converted. Advised format is 2048,384
        beats_per_minute (int, optional): Sets beats per minute of file. Defaults to 120.
        tick_per_beat (int, optional): Sets amount of ticks(numbers) per beat. Defaults to 128.
        midi_name (str, optional): header name within midi file. Defaults to 'placeholder'.

    Returns:
        str: string of JSON information about the input array that can be read by tone.js
    """    
    
    ####Header information####
    header = {
    "keySignatures": [],
    "meta": [],
    "name": midi_name,
    "ppq": tick_per_beat,
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

    track_length = tick_per_beat*4*4

    #Convert input track to 1 bar of 4 ticks per beat for demo
    target_length = tick_per_beat * 4
    bar_temp = input_array[:512,:]

    bar_shape = np.shape(bar_temp)
    bar_shortened = np.empty((target_length,bar_shape[1]))

    for i in range(bar_shape[1]):
        row = bar_temp[:,i]
        new_audio_row = row[:8 * target_length].reshape((8, target_length)).mean(axis=0)
        bar_shortened[:,i] = new_audio_row

    bar_shortened[bar_shortened < 0.02] = 0



    #Seperate tracks 


    drum_lookup = drum_dict()
    rhythm_lookup = rhythm_dict()
    lead_lookup = lead_dict()

    drum_raw = input_array[:,0:len(drum_lookup)]
    rhythm_raw = input_array[:,len(drum_lookup):(len(drum_lookup)+len(rhythm_lookup))]
    lead_raw = input_array[:,(len(drum_lookup)+len(rhythm_lookup)):(len(drum_lookup)+len(rhythm_lookup)+len(lead_lookup))]

    drum_new = np.empty((track_length,128))
    rhythm_new = np.empty((track_length,128))
    lead_new = np.empty((track_length,128))

    for i in range(0,len(drum_lookup)):
        column = drum_raw[:,i] 
        midi_code = drum_lookup[i]
        drum_new[:,midi_code] = column

    for i in range(0,len(rhythm_lookup)):
        column = rhythm_raw[:,i] 
        midi_code = rhythm_lookup[i]
        rhythm_new[:,midi_code] = column
    
    for i in range(0,len(lead_lookup)):
        column = lead_raw[:,i] 
        midi_code = lead_lookup[i]
        lead_new[:,midi_code] = column

    


    list_of_tracks = [drum_new, rhythm_new, lead_new]
    track_names = [['drums',10], ['rhythm',11],['lead',12]]


    # create events list for each instrument
    events = []
    for track in range(0,len(list_of_tracks)):
        temp_events = extract_events(list_of_tracks[track], beats_per_minute=beats_per_minute, tick_per_beat=tick_per_beat)
        events.append(temp_events)

    tracks = []

    for track in range(0,len(events)):
        if len(events[track]) > 1:
            temp_track = create_track(input_array = list_of_tracks[track], input_track_events = events[track], instrument= track_names[track][0],channel= track_names[track][1])
            tracks.append(temp_track)

    output_dictionary = {"header":header, 
                        "tracks":tracks}

    output_json = json.dumps(output_dictionary)


    return output_json

#############################################################################################
#############################################################################################