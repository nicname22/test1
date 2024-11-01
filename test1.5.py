import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import scipy.fftpack

# F5, D5, A5, D5, F5, D5, A5, D5, F5, C5, A5, C5, F5, C5, A5, C5, 
# E5, C#5, A4, C#5, E5, C#5, A4, C#5, E5, C#5, A4, C#5, E5, C#5, A4, C#5
notes = [
    ['C0', 16.35, []],
    ['C#0', 17.32, []],
    ['D0', 18.35, []],
    ['D#0', 19.45, []],
    ['E0', 20.60, []],
    ['F0', 21.83, []],
    ['F#0', 23.12, []],
    ['G0', 24.50, []],
    ['G#0', 25.96, []],
    ['A0', 27.50, []],
    ['A#0', 29.14, []],
    ['B0', 30.87, []],
    ['C1', 32.70, []],
    ['C#1', 34.65, []],
    ['D1', 36.71, []],
    ['D#1', 38.89, []],
    ['E1', 41.20, []],
    ['F1', 43.65, []],
    ['F#1', 46.25, []],
    ['G1', 49.00, []],
    ['G#1', 51.91, []],
    ['A1', 55.00, []],
    ['A#1', 58.27, []],
    ['B1', 61.74, []],
    ['C2', 65.41, []],
    ['C#2', 69.30, []],
    ['D2', 73.42, []],
    ['D#2', 77.78, []],
    ['E2', 82.41, []],
    ['F2', 87.31, []],
    ['F#2', 92.50, []],
    ['G2', 98.00, []],
    ['G#2', 103.83, []],
    ['A2', 110.00, []],
    ['A#2', 116.54, []],
    ['B2', 123.47, []],
    ['C3', 130.81, []],
    ['C#3', 138.59, []],
    ['D3', 146.83, []],
    ['D#3', 155.56, []],
    ['E3', 164.81, []],
    ['F3', 174.61, []],
    ['F#3', 185.00, []],
    ['G3', 196.00, []],
    ['G#3', 207.65, []],
    ['A3', 220.00, []],
    ['A#3', 233.08, []],
    ['B3', 246.94, []],
    ['C4', 261.63, []],
    ['C#4', 277.18, []],
    ['D4', 293.66, []],
    ['D#4', 311.13, []],
    ['E4', 329.63, []],
    ['F4', 349.23, []],
    ['F#4', 369.99, []],
    ['G4', 392.00, []],
    ['G#4', 415.30, []],
    ['A4', 440.00, []],
    ['A#4', 466.16, []],
    ['B4', 493.88, []],
    ['C5', 523.25, []],
    ['C#5', 554.37, []],
    ['D5', 587.33, []],
    ['D#5', 622.25, []],
    ['E5', 659.25, []],
    ['F5', 698.46, []],
    ['F#5', 739.99, []],
    ['G5', 783.99, []],
    ['G#5', 830.61, []],
    ['A5', 880.00, []],
    ['A#5', 932.33, []],
    ['B5', 987.77, []],
    ['C6', 1046.50, []],
    ['C#6', 1108.73, []],
    ['D6', 1174.66, []],
    ['D#6', 1244.51, []],
    ['E6', 1318.51, []],
    ['F6', 1396.91, []],
    ['F#6', 1479.98, []],
    ['G6', 1567.98, []],
    ['G#6', 1661.22, []],
    ['A6', 1760.00, []],
    ['A#6', 1864.66, []],
    ['B6', 1975.53, []],
    ['C7', 2093.00, []],
    ['C#7', 2217.46, []],
    ['D7', 2349.32, []],
    ['D#7', 2489.02, []],
    ['E7', 2637.02, []],
    ['F7', 2793.83, []],
    ['F#7', 2959.96, []],
    ['G7', 3135.96, []],
    ['G#7', 3322.44, []],
    ['A7', 3520.00, []],
    ['A#7', 3729.31, []],
    ['B7', 3951.07, []],
    ['C8', 4186.01, []],
    ['C#8', 4434.92, []],
    ['D8', 4698.63, []],
    ['D#8', 4978.03, []],
    ['E8', 5274.04, []],
    ['F8', 5587.65, []],
    ['F#8', 5919.91, []],
    ['G8', 6271.93, []],
    ['G#8', 6644.88, []],
    ['A8', 7040.00, []],
    ['A#8', 7458.62, []],
    ['B8', 7902.13, []],
]

FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 88200
CHUNK = 5012
RECORD_SECONDS = 40

# cd=0.25,nr=0.4
chunk_duration = 0.1 # Analyze the signal in 1-second chunks
note_range = 0.4 # Detect notes based on amplitude threshold
# tick = 0.05
# chunk_duration = 0.05
# note_range = 0.4

def normalizeNote(note):
    return note[:-1]

# Find the closest musical note for a given frequency
def getNote(frequency):
    for noteData in notes:
        noteFreq = noteData[1]
        if noteFreq * 0.986 <= frequency <= noteFreq * 1.015:
            return noteData[0]  # Return note and octave
    return ''

def detect_in_chunks(wav_file):
    fileSampleRate, signal = wavfile.read(wav_file)

    # Convert stereo to mono if necessary
    if len(signal.shape) == 2:
        signal = signal.sum(axis=1) / 2
        
    N = signal.shape[0]  # Total number of samples
    samples_per_chunk = int(fileSampleRate * chunk_duration)
    global_notes_sequence = []  # To store all notes across chunks
    # Split the signal into chunks
    for start_sample in range(0, N, samples_per_chunk):
        end_sample = start_sample + samples_per_chunk
        if end_sample > N:
            break

        chunk_signal = signal[start_sample:end_sample]

        # Perform FFT on the chunk
        fft = abs(scipy.fft.fft(chunk_signal))
        fftOneSide = fft[:len(fft)//2]
        fftFrequencies = scipy.fftpack.fftfreq(len(chunk_signal), d=1.0/fileSampleRate)
        fftFrequenciesOneSide = fftFrequencies[:len(fft)//2]

        normalizedAbsoluteValues = abs(fftOneSide) / np.linalg.norm(abs(fftOneSide))

        detected_notes = []
        # Analyze frequencies in the chunk
        for frequencyIndex in range(0, len(fftFrequenciesOneSide)):
            if 110 <= fftFrequenciesOneSide[frequencyIndex] <= 8200:  # Focus on musical range
                if normalizedAbsoluteValues[frequencyIndex] > note_range:
                    note = getNote(fftFrequenciesOneSide[frequencyIndex])
                    if note:
                        normalized_note = normalizeNote(note)
                        detected_notes.append(normalized_note)

        # If notes were detected in chunk, merge
        if detected_notes:
            merged_notes = set(detected_notes)  # Remove duplicates within the chunk
            for note in merged_notes:
                if not global_notes_sequence or global_notes_sequence[-1] != note:
                    global_notes_sequence.append(note)

    # Output the simplified sequence of detected notes
    print("Final merged notes sequence:", ', '.join(global_notes_sequence))

# Example usage
detect_in_chunks("GFTest1.wav")
# detect_in_chunks("megtest.wav")