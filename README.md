# 🎧 Air DJ 3.0

> Control your music with just your hands – speed, pitch, and volume, all in real-time!

Air DJ 3.0 is an **interactive Python application** that lets you mix your favorite songs using your webcam. Using **hand gestures**, you can dynamically control the **speed, pitch, and volume** of any audio track. Perfect for experimenting, performing, or just having fun with your favorite tracks!  

---

## 🚀 Features

- **Real-time gesture-based control** using MediaPipe.
- **Speed control**: Right-hand distance (thumb ↔ index) adjusts playback speed.
- **Pitch control**: Left-hand distance (thumb ↔ index) changes pitch in semitones.
- **Volume control**: Distance between both hands adjusts volume smoothly.
- **Continuous playback** with minimal stutter.
- **Support for any `.mp3` file**.
- **Cross-platform**: Works on Windows (tested) — can be adapted to Linux/Mac.

---

## 🎶 How to Use

1. **Clone the repo**:

```bash
git clone https://github.com/YourUsername/AirDJ.git
cd AirDJ


2. **install dependencies**
pip install -r requirements.txt


3. **Place your .mp3 file in the project folder and rename it to song.mp3 (or change the FILENAME variable in main.py)**

4. **run the app**
python main.py

5. **Use your hands to control the song**

Right-hand thumb ↔ index → Speed
Left-hand thumb ↔ index → Pitch
Distance between hands → Volume
Press q to quit


🛠 Tech Stack

Python 3.10+
OpenCV – webcam input and gesture detection
MediaPipe – hand tracking
Librosa + RubberBand – real-time pitch and speed adjustments
SoundDevice – low-latency audio output
NumPy – math and smoothing


🔧 Notes
Smaller audio chunks are used to reduce latency, but powerful CPU helps for smooth performance
Smooth transitions are applied to speed, pitch, and volume to make the experience natural
Tested on Windows, should work on other platforms with minimal adjustments


💡 License
This project is open-source, feel free to fork, contribute, or remix!

Made with love by youcef chalbi