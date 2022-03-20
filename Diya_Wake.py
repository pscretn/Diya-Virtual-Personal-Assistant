from Diya_Speak import say
from Diya_Listen import wakeup
from Diya import call_diya

while True:
    print("Wake me up by calling Diya")
    text = wakeup()
    if "diya" in text:
        call_diya()
        