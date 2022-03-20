import json
import numpy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from NeuralNetwork_of_Diya import word_list, tokenize, stem
from Diya_Processing import NeuralNetwork

with open('intents.json', 'r') as f:
    intents = json.load(f)

All_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)

    for pattern in intent['patterns']:
        #print(pattern)
        wds = tokenize(pattern)
        All_words.extend(wds)
        xy.append((wds, tag))

ignore_words = ['?', '.', ',', '!',"/"]
All_words = [stem(wd) for wd in All_words if wd not in ignore_words]
All_words = sorted(set(All_words))
tags = sorted(set(tags))

x_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = word_list(pattern_sentence, All_words)
    x_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

x_train = numpy.array(x_train)
y_train = numpy.array(y_train)
num_approx = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(x_train[0])
hidden_size = 8
output_size = len(tags)
print("training the model.......")

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples=len(x_train)
        self.x_data=x_train
        self.y_data=y_train
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,num_workers=0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNetwork(input_size, hidden_size, output_size).to(device=device)
criterion = nn.CrossEntropyLoss()
optmizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for aprox in range (num_approx):
    for (words, labels) in train_loader:
        words = words.to(device=device)
        labels = labels.to( dtype=torch.long).to(device=device)
        output = model(words)
        loss = criterion(output, labels)
        optmizer.zero_grad()
        loss.backward()
        optmizer.step()

    if (aprox+1)%100 == 0:
        print(f'Approximation [{aprox+1}/{num_approx}],Loss: {loss.item():.4f}')

print(f'Final Loss: {loss.item():.4f}')

data = {
    "model_state":model.state_dict(),"input_size":input_size,"hidden_size":hidden_size,"output_size":output_size,"All_words":All_words,"tags":tags
}

FILE = "TrainedData.pth"
torch.save(data, FILE)
print(f"Model Trained and File is Saved to {FILE}") 