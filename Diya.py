import torch
import random
import json
from NeuralNetwork_of_Diya import word_list , tokenize 
from Diya_Processing import NeuralNetwork
from Diya_Speak import say
from Tasks_of_Diya import  NonInputExecution , InputExecution
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "TrainedData.pth"
data = torch.load(FILE)
input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
All_words = data['All_words']
tags = data['tags']
model_state = data['model_state']
model = NeuralNetwork(input_size,hidden_size,output_size).to(device)
model.load_state_dict(model_state)
model.eval()

Name = "Diya"
from Diya_Listen import listen

def call_diya():
    text = listen()
    result = str(text).lower()
    text = tokenize(text)
    X = word_list(text,All_words)
    X = X.reshape(1,X.shape[0])
    X = torch.from_numpy(X).to(device)
    output = model(X)
    _,prediction = torch.max(output,dim=1)
    tag = tags[prediction.item()]
    probs = torch.softmax(output,dim=1)
    prob = probs[0][prediction.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if intent['tag'] == tag:
                reply = random.choice(intent['responses'])

                if "time" in reply:
                    NonInputExecution(reply)
                elif "date" in reply:
                    NonInputExecution(reply)
                elif "day" in reply:
                    NonInputExecution(reply)
                elif "wikipedia" in reply:
                    InputExecution(reply,result)
                elif "google" in reply:
                    InputExecution(reply,result)
                else:
                    say(reply)
    else:
        say("I am not sure what you mean")
    