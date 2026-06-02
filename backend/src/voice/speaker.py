import pyttsx3

# Initialize speech engine
engine = pyttsx3.init()

# Speech speed
engine.setProperty("rate", 150)

def speak(text):
    """
    Convert text to speech.
    """

    if text.strip():

        print(f"Speaking: {text}")

        engine.say(text)

        engine.runAndWait()