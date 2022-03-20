from datetime import datetime
from Diya_Speak import say
import wikipedia
import pywhatkit

def Time():
    time = datetime.now().strftime("%I:%M %p")
    say(time)

def Date():
    date = datetime.now().strftime("%d %B %Y")
    say(date)
def Day():
    day = datetime.now().strftime("%A")
    say(day)

def NonInputExecution(query):
    query = str(query)
    if "time" in query:
        Time()
    elif "date" in query:
        Date()
    elif "day" in query:
        Day()



def InputExecution(tag,query):
    if 'wikipedia' in tag:
        name = str(query).replace("who is","").replace("about","").replace("what is","").replace("where is","").replace("when is","").replace("wikipedia","")
        result = wikipedia.summary(name,sentences=2)
        say(result)
    elif 'google' in tag:
        query=str(query).replace("google","")
        query=query.replace("search","")
        pywhatkit.search(query)