import speech_recognition as sr
from pydub import AudioSegment
from pydub.playback import play
startbeep =  AudioSegment.from_wav("sounds\startbeep.wav")
stopbeep =  AudioSegment.from_wav("sounds\endbeep.wav")

def listen():
    try:
        record = sr.Recognizer()
        with sr.Microphone() as mic:
            record.adjust_for_ambient_noise(mic, duration=0.2)
            print("Listening...")
            play(startbeep)
            audio = record.listen(mic,0,3)
            print("Recognizing...")
            query = record.recognize_google(audio, language="en-in")
            play(stopbeep)
            print(f"You Said : {query} ")
    
    except:
        return ""

    return query.lower()

def wakeup():
    
    try:
        record  = sr.Recognizer()
        with sr.Microphone() as mic:
            record.adjust_for_ambient_noise(mic, duration=0.2)
            audio = record.listen(mic,0,2)
            sentence = record.recognize_google(audio, language="en-in")
    except:
        return ""
    return sentence.lower()
