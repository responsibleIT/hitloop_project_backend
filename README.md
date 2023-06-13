# Hitloop Backend

This repo is for the API of Hitloop. This has two main features: Hosting of the Audio samples and generating MIDI patterns.



The API is created using server.py, this launches a REST-API where get requests can be used to retrieve the files.

The code needed to transform arrays into json files for Tone.JS, are created in the midi_functions.py file.



## Samples

The API hosts three sample packs.

Samples folder = Handcreated samples for backup

Samples_a = Randomly created samples

Samples_b = AI created samples

These were all created using the system described in our paper. 



## MIDI generation

Using the model 'Generator.h5' the API can generated MIDI files. This generator is the generator from the GAN 1, as described in our paper. 



