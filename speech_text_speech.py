import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import time
import whisper
import pyttsx3

def record_audio(seconds=5, samplerate=16000, channels=1):
    print("Recording... Speak now!")
    audio = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=channels, dtype="float32")
    sd.wait()
    audio = np.squeeze(audio)
    path = tempfile.gettempdir() + f"/record_{int(time.time())}.wav"
    sf.write(path, audio, samplerate)
    return path

def speech_to_text():
    path = record_audio()
    model = whisper.load_model("tiny") 
    print("Transcribing...")
    result = model.transcribe(path, fp16=False, language="en") 
    text = result.get("text", "").strip()
    print("You said:", text)

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 180)
    engine.say(text)
    engine.runAndWait()

def main():
    while True:
        print("\n1) Text to Speech")
        print("2) Speech to Text")
        print("3) Exit")
        choice = input("Choose option: ").strip()

        if choice == "1":
            text = input("Enter text: ").strip()
            if text:
                text_to_speech(text)
        elif choice == "2":
            speech_to_text()
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice, try again.")

if _name_ == "_main_":
    main()
