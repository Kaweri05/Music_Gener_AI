import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from music21 import converter, instrument, note, stream, chord
import random

st.set_page_config(page_title="AI Music Generator", page_icon=":musical_note:")

st.title("AI Music Generator")
st.write("Generate music using a trained LSTM model.")

@st.cache_resource
def load_artifacts():
    """Loads the trained model and notes data."""
    try:
        model = load_model("music_model.keras")
        with open("notes.pkl", "rb") as f:
            notes = pickle.load(f)
        pitchnames = sorted(list(set(notes)))
        return model, notes, pitchnames
    except FileNotFoundError:
        st.error("Model files (music_model.keras or notes.pkl) not found. Please ensure they are uploaded.")
        return None, None, None

model, notes, pitchnames = load_artifacts()

if model and notes and pitchnames:
    sequence_length = 50 # This should match the sequence_length used during training

    note_to_int = dict(
        (note_val, number)
        for number, note_val in enumerate(pitchnames)
    )
    int_to_note = dict(
        (number, note_val)
        for number, note_val in enumerate(pitchnames)
    )

    def generate_music(model, notes, pitchnames, sequence_length):
        """Generates a MIDI sequence based on the trained model."""
        start = random.randint(0, len(notes) - sequence_length - 1)
        if start < 0: # Handle cases where notes might be too short
            st.warning("Not enough notes to generate a sequence. Please upload more MIDI data.")
            return None

        pattern_raw = notes[start : start + sequence_length]
        pattern_int = [note_to_int[n] for n in pattern_raw]

        prediction_output = []

        # Generate 200 notes
        for _ in range(200):
            prediction_input = np.reshape(pattern_int, (1, sequence_length, 1))
            prediction_input = prediction_input / float(len(pitchnames))

            prediction = model.predict(prediction_input, verbose=0)
            index = np.argmax(prediction)
            result = int_to_note[index]
            prediction_output.append(result)

            pattern_int.append(index)
            pattern_int = pattern_int[1:] # Shift window

        output_notes = []
        offset = 0
        for pattern_val in prediction_output:
            try:
                # If the pattern is a chord
                if ('.' in pattern_val) or pattern_val.isdigit():
                    notes_in_chord = [int(char) for char in pattern_val.split('.')]
                    new_chord = chord.Chord()
                    for current_note in notes_in_chord:
                        n = note.Note(pitchnames[current_note]) # Assuming pitchnames can map back to actual note names
                        new_chord.add(n)
                    new_chord.offset = offset
                    output_notes.append(new_chord)
                # If the pattern is a single note
                else:
                    new_note = note.Note(pattern_val)
                    new_note.offset = offset
                    new_note.storedInstrument = instrument.Piano()
                    output_notes.append(new_note)
            except Exception as e:
                st.warning(f"Could not parse note/chord: {pattern_val}. Error: {e}")
                # Fallback to a default note if parsing fails
                new_note = note.Note("C4")
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)
            offset += 0.5

        midi_stream = stream.Stream(output_notes)
        midi_stream.write('midi', fp='generated_music.mid')
        return 'generated_music.mid'

    if st.button("Generate Music"):
        with st.spinner("Generating music..."):
            midi_file_path = generate_music(model, notes, pitchnames, sequence_length)
            if midi_file_path:
                st.success("Music Generated!")
                with open(midi_file_path, "rb") as file:
                    st.download_button(
                        label="Download Generated Music (MIDI)",
                        data=file,
                        file_name="generated_music.mid",
                        mime="audio/midi"
                    )
            else:
                st.error("Failed to generate music.")
else:
    st.warning("Model and data not loaded. Please check the files and run the previous cells.")
