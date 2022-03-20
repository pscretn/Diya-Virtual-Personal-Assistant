import pyttsx3 as pyt

def say(text):
    engine = pyt.init()
    voices = engine.getProperty('voices')
    rate = engine.getProperty('rate')
    engine.setProperty('voice',voices[1].id)
    engine.setProperty('rate',140)
    engine.say(text)
    engine.runAndWait()
