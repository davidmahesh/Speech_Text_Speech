import sounddevice as sd      # for recording audio from microphone
import soundfile as sf        # for saving recorded audio
import numpy as np            # for handling audio data
import tempfile               # for creating a temporary file path
import time                   # for unique filenames
import whisper                # OpenAI Whisper model for speech-to-text
import pyttsx3                # for text-to-speech (offline)

# Function to record the audio

def record_audio(seconds=5, samplerate=16000, channels=1):
    print("Recording... Speak now!")
    # start recording: number of samples = seconds * samplerate
    audio = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=channels, dtype="float32")
    sd.wait()  # waits until the recording completes
    
    # removes the extra dimension if only one channel
    audio = np.squeeze(audio)
    
    # creates temporary file path with current time
    path = tempfile.gettempdir() + f"/record_{int(time.time())}.wav"
    
    # save audio data into wav file
    sf.write(path, audio, samplerate)
    
    return path  # returns the file path

# Function: Speech to Text
def speech_to_text():
    path = record_audio()  # records the voice first
    
    # load Whisper model (tiny --> fastest but less accurate)
    model = whisper.load_model("tiny")
    
    print("Transcribing...")
    # transcribes the audio to English text
    result = model.transcribe(path, fp16=False, language="en")
    
    # extracts only text from result
    text = result.get("text", "").strip()
    print("You said:", text)

# Function: Text to Speech
def text_to_speech(text):
    engine = pyttsx3.init()        # initialize text-to-speech engine
    engine.setProperty("rate", 180) # set speaking speed (default 200)
    engine.say(text)                # gives text to engine
    engine.runAndWait()             # speaks the text

# Main function with the menu
def main():
    while True:
        # menu options for the user
        print("\n1) Text to Speech")
        print("2) Speech to Text")
        print("3) Exit")
        
        choice = input("Choose option: ").strip()  # takes the user choice

        if choice == "1":
            # asks user for text and convert to the voice
            text = input("Enter text: ").strip()
            if text:
                text_to_speech(text)
        
        elif choice == "2":
            # record voice and convert to text
            speech_to_text()
        
        elif choice == "3":
            print("Exiting...")  # exits the program
            break
        
        else:
            print("Invalid choice, try again.")  # wrong input

# Runs the main function if this file is executed directly
if __name__ == "__main__":
    main()
