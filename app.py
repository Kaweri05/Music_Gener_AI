
import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from music21 import instrument, note, stream, chord
import random

# Page configuration
st.set_page_config(
    page_title="AI Music Generator",
    page_icon="🎵"
)

# App title
st.title("AI Music Generator")
st.write("Generate music using a trained LSTM model.")


@st.cache_resource
def load_artifacts():
    """Loads the trained model and notes data."""

    try:
        model = load_model(
            "fixed_music_model.keras",
            compile=False
        )

        with open("notes.pkl", "rb") as f:
            notes = pickle.load(f)

        pitchnames = sorted(set(notes))

        return model, notes, pitchnames

    except FileNotFoundError:
        st.error(
            "Model files not found."
        )
        return None, None, None

    except Exception as e:
        st.error(
            f"Error loading model: {e}"
        )
        return None, None, None


    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None


# Load model and data
model, notes, pitchnames = load_artifacts()

if (
    model is not None
    and notes is not None
    and pitchnames is not None
):

    sequence_length = 50

    note_to_int = {
        note_val: number
        for number, note_val in enumerate(
            pitchnames
        )
    }

    int_to_note = {
        number: note_val
        for number, note_val in enumerate(
            pitchnames
        )
    }

    def generate_music(
        model,
        notes,
        pitchnames,
        sequence_length
    ):

        if len(notes) <= sequence_length:
            st.warning(
                "Not enough notes "
                "available to generate music."
            )
            return None

        start = random.randint(
            0,
            len(notes)
            - sequence_length
            - 1
        )

        pattern_raw = notes[
            start:start + sequence_length
        ]

        pattern_int = [
            note_to_int[n]
            for n in pattern_raw
        ]

        prediction_output = []

        for _ in range(200):

            prediction_input = np.reshape(
                pattern_int,
                (1, sequence_length, 1)
            )

            prediction_input = (
                prediction_input /
                float(len(pitchnames))
            )

            prediction = model.predict(
                prediction_input,
                verbose=0
            )

            index = np.argmax(
                prediction
            )

            result = int_to_note[
                index
            ]

            prediction_output.append(
                result
            )

            pattern_int.append(index)
            pattern_int = pattern_int[1:]

        output_notes = []
        offset = 0

        for pattern_val in prediction_output:

            try:
                # Chord handling
                if "." in str(pattern_val):

                    notes_in_chord = [
                        int(n)
                        for n in pattern_val.split(".")
                    ]

                    chord_notes = []

                    for current_note in notes_in_chord:

                        new_note = note.Note(
                            pitchnames[current_note]
                        )

                        new_note.storedInstrument = (
                            instrument.Piano()
                        )

                        chord_notes.append(
                            new_note
                        )

                    new_chord = chord.Chord(
                        chord_notes
                    )

                    new_chord.offset = (
                        offset
                    )

                    output_notes.append(
                        new_chord
                    )

                else:
                    new_note = note.Note(
                        str(pattern_val)
                    )

                    new_note.offset = (
                        offset
                    )

                    new_note.storedInstrument = (
                        instrument.Piano()
                    )

                    output_notes.append(
                        new_note
                    )

            except Exception as e:
                st.warning(
                    f"Could not parse "
                    f"{pattern_val}: {e}"
                )

            offset += 0.5

        midi_stream = stream.Stream(
            output_notes
        )

        midi_stream.write(
            "midi",
            fp="generated_music.mid"
        )

        return "generated_music.mid"

    # Button
    if st.button(
        "Generate Music"
    ):

        with st.spinner(
            "Generating music..."
        ):

            midi_file_path = (
                generate_music(
                    model,
                    notes,
                    pitchnames,
                    sequence_length
                )
            )

            if midi_file_path:

                st.success(
                    "Music Generated!"
                )

                with open(
                    midi_file_path,
                    "rb"
                ) as file:

                    st.download_button(
                        label="Download Generated Music (MIDI)",
                        data=file,
                        file_name="generated_music.mid",
                        mime="audio/midi"
                    )

            else:
                st.error(
                    "Failed to generate music."
                )

else:
    st.warning(
        "Model and data not loaded."
    )

