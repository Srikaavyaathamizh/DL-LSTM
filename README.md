# Experiment 6 : Developing a Deep Learning Model for NER using LSTM
## NAME : Srikaavyaa T
## REGISTRATION NUMBER : 212223230214

## AIM
To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset
The goal of this project is to develop a Long Short-Term Memory (LSTM) based model to perform Named Entity Recognition (NER) on text data. The model will identify and classify words in a sentence into predefined categories such as Person, Organization, Location, Date, Time, Money, Percent, Facility, and GPE. Using the NER dataset containing Word, POS, and Tag columns, the model will be trained to recognize entities and predict the correct tags for unseen text.


<img width="618" height="763" alt="DATASET 6 SS1" src="https://github.com/user-attachments/assets/900bc91d-0061-475b-bab5-4cdedc7e811e" />

## DESIGN STEPS
### STEP 1: 

Load data, create word/tag mappings, and group sentences.

### STEP 2: 
Convert sentences to index sequences, pad to fixed length, and split into training/testing sets.


### STEP 3: 
Define dataset and DataLoader for batching.


### STEP 4: 

Build a bidirectional LSTM model for sequence tagging.

### STEP 5: 
Train the model over multiple epochs, tracking loss


### STEP 6: 

Evaluate model accuracy, plot loss curves, and visualize predictions on a sample.





## PROGRAM

### Name: Srikaavyaa T

### Register Number: 212223230214

```
class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=50, hidden_dim=100):
        super(BiLSTMTagger, self).__init__()
        self.embedding=nn.Embedding(vocab_size, embedding_dim)
        self.dropout=nn.Dropout(0.1)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, tagset_size)
    def forward(self,x):
        x=self.embedding(x)
        x=self.dropout(x)
        x,_=self.lstm(x)
        return self.fc(x)


model =BiLSTMTagger(len(word2idx)+1,len(tag2idx)).to(device)
loss_fn =nn.CrossEntropyLoss()
optimizer =torch.optim.Adam(model.parameters(),lr=0.001)


# Training and Evaluation Functions
def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3):
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        total_loss=0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = loss_fn(outputs.view(-1, len(tag2idx)), labels.view(-1))
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
        train_losses.append(total_loss)
        model.eval()
        val_loss=0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids)
                loss = loss_fn(outputs.view(-1, len(tag2idx)), labels.view(-1))
                val_loss+=loss.item()
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss}, Val Loss: {val_loss}")
    return train_losses, val_losses

```

### OUTPUT

## Loss Vs Epoch Plot

<img width="498" height="423" alt="image" src="https://github.com/user-attachments/assets/9671c200-c9aa-4311-87d9-3720b741ed0f" />





### Sample Text Prediction
<img width="283" height="347" alt="image" src="https://github.com/user-attachments/assets/08542817-12d3-445a-ba79-de9c062ef958" />


## RESULT
Thus, an LSTM-based model for recognizing the named entities in the text has been developed successfully.
