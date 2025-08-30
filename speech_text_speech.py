# Import required libraries
import sounddevice as sd      # For recording audio from microphone
import soundfile as sf        # For saving recorded audio 
import numpy as np            # For handling and processing audio data
import tempfile               # For creating a temporary file path
import time                   # For generating unique filenames using timestamps
import whisper                # OpenAI Whisper model for speech-to-text conversion
import pyttsx3                # For text-to-speech (offline, works without internet)

# ----------------------------
# Function: Record Audio
# ----------------------------
def record_audio(seconds=5, samplerate=16000, channels=1):
    """
    Records audio using the microphone 
    
    """
    print("Recording... Speak now!")  # Notify user that recording has started
    
    # Start recording audio
    # Total samples = seconds * samplerate
    audio = sd.rec(
        int(seconds * samplerate),  # Number of samples to record
        samplerate=samplerate,      # Sampling rate (Hz)
        channels=channels,          # Number of channels (1 = mono, 2 = stereo)
        dtype="float32"             # Data type of recorded samples
    )
    
    sd.wait()  # Wait until recording finishes before proceeding
    
    # If only one channel is recorded, remove the extra dimension
    audio = np.squeeze(audio)
    
    # Generate a temporary file path using timestamp (unique filename)
    path = tempfile.gettempdir() + f"/record_{int(time.time())}.wav"
    
    # Save recorded audio data
    sf.write(path, audio, samplerate)
    
    return path  # Return the file path of the recorded audio

# ----------------------------
# Function: Speech-to-Text
# ----------------------------
def speech_to_text():
    """
    Records speech from microphone and transcribes it into text using Whisper.
    """
    path = record_audio()  # Record audio and get its file path
    
    # Load Whisper model ("tiny" is the fastest, but less accurate)
    model = whisper.load_model("tiny")
    
    print("Transcribing...")  # Notify user that transcription has started
    
    # Transcribe the recorded audio to English text
    # fp16=False ensures compatibility with CPUs (no GPU float16 issues)
    result = model.transcribe(path, fp16=False, language="en")
    
    # Extract only the transcribed text (remove extra details like timestamps)
    text = result.get("text", "").strip()
    
    # Display the recognized speech
    print("You said:", text)

# ----------------------------
# Function: Text-to-Speech
# ----------------------------
def text_to_speech(text):
    """
    Converts input text into speech using pyttsx3 (offline TTS).
    """
    engine = pyttsx3.init()         # Initialize text-to-speech engine
    
    engine.setProperty("rate", 180) # Set speaking speed (default ~200 words/min)
    
    engine.say(text)                # Queue the text for speaking
    
    engine.runAndWait()             # Speak the text out loud

# ----------------------------
# Main Program Menu
# ----------------------------
def main():
    """
    Main function to provide user menu for Text-to-Speech and Speech-to-Text.
    """
    while True:  # Infinite loop until user chooses Exit
        # Display menu options
        print("\n1) Text to Speech")
        print("2) Speech to Text")
        print("3) Exit")
        
        # Get user choice
        choice = input("Choose option: ").strip()
        
        if choice == "1":
            # Option 1: Text-to-Speech
            text = input("Enter text: ").strip()  # Ask user for input text
            if text:  # If user entered some text
                text_to_speech(text)  # Convert text to speech
        
        elif choice == "2":
            # Option 2: Speech-to-Text
            speech_to_text()  # Record audio and transcribe it
        
        elif choice == "3":
            # Option 3: Exit program
            print("Exiting...")
            break  # Exit the loop, end program
        
        else:
            # If user enters invalid choice
            print("Invalid choice, try again.")

# ----------------------------
# Run program if executed directly
# ----------------------------
if __name__ == "__main__":
    main()  # Start main function
