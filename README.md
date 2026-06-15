# 🎵 AI Music Generator

An AI-powered music generation system that creates MIDI music using a trained **LSTM (Long Short-Term Memory)** deep learning model. The application is built with **TensorFlow/Keras** and deployed using **Streamlit**.

---

## 📌 Project Overview

The **AI Music Generator** generates new musical sequences based on patterns learned from MIDI music datasets. It uses an **LSTM neural network** to understand musical note sequences and predict the next notes to create original music.

Users can generate AI-based music and download the output as a **MIDI file**.

---

## 🚀 Features

* 🎼 Generate music using a trained LSTM model
* 🎹 MIDI file generation
* 📥 Download generated music
* 🌐 Simple and interactive Streamlit UI
* 🧠 Deep Learning-based music prediction

---

## 🛠️ Tech Stack

### Programming Language

* Python

### Libraries & Frameworks

* Streamlit
* TensorFlow
* Keras
* NumPy
* Music21
* Pretty MIDI
* Pickle

### Deep Learning Model

* **LSTM (Long Short-Term Memory)**

---

## 📂 Project Structure

```txt
Music_Gener_AI/
│── app.py
│── requirements.txt
│── runtime.txt
│── fixed_music_model.keras
│── notes.pkl
│── generated_music.mid
│── README.md
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/Kaweri05/Music_Gener_AI.git
cd Music_Gener_AI
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## 📊 Working Process

1. Load trained LSTM model
2. Load musical notes dataset (`notes.pkl`)
3. Generate note patterns using prediction
4. Convert predicted notes into MIDI format
5. Download generated music file

---

## 📸 Screenshots

Add your project screenshots here.

Example:

```md
![Home Page](images/home.png)
```

---

## 🎯 Applications

* AI Music Composition
* Music Recommendation Systems
* Background Music Generation
* Learning Deep Learning in Music

---

## 🔮 Future Enhancements

* Add multiple instrument support
* Generate music in different genres
* Add real-time audio playback
* Improve UI design
* Add tempo and style control

---

## 👨‍💻 Author

**Kaweri Harinkhede**
Computer Engineering Student
Passionate about AI, Machine Learning, and Web Development

---

## 📜 License

This project is developed for educational and learning purposes.
