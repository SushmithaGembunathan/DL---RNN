# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Problem Statement and Dataset
To design and implement a Recurrent Neural Network (RNN) that learns temporal patterns from historical stock closing prices and predicts future stock prices based on past trends.
The dataset consists of historical stock market data containing daily closing prices of a selected company, which is preprocessed through normalization and sequence generation before being used for training and testing the RNN model.


<img width="754" height="172" alt="image" src="https://github.com/user-attachments/assets/3439e4da-7c1e-463c-a6cd-2fca6f5e7516" />


## DESIGN STEPS
## STEP 1:
Collect historical stock closing price data and perform preprocessing such as normalization and sequence creation.

## STEP 2:
Split the dataset into training and testing sets and convert them into PyTorch tensors and DataLoader format.

## STEP 3:
Design an RNN model using input, hidden, and output layers suitable for time-series prediction.

## STEP 4:
Define the loss function (Mean Squared Error) and optimizer (Adam) for training the model.

## STEP 5:
Train the RNN model over multiple epochs while updating weights using backpropagation through time.

## STEP 6:
Evaluate the trained model by plotting training loss and comparing predicted stock prices with actual prices.


## PROGRAM

### Name: Sushmitha Gembunathan 

### Register Number:212224040342

```python
# Define RNN Model
class RNNModel(nn.Module):
  def __init__(self,input_size=1,hidden_size=64,num_layers=2,output_size=1):
    super(RNNModel,self).__init__()
    self.rnn=nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
    self.fc=nn.Linear(hidden_size,output_size)

  def forward(self,x):
    out,_=self.rnn(x)
    out=self.fc(out[:,-1,:])
    return out




# Train the Model

def train_model(model,train_loader,criterion,optimizer,epochs=20):
  train_losses=[]
  model.train()
  for epoch in range(epochs):
    total_loss=0
    for x_batch,y_batch in train_loader:
      x_batch,y_batch=x_batch.to(device),y_batch.to(device)
      optimizer.zero_grad()
      outputs=model(x_batch)
      loss=criterion(outputs,y_batch)
      loss.backward()
      optimizer.step()
      total_loss+=loss.item()
    train_losses.append(total_loss/len(train_loader))
    print(f'Epoch {epoch+1}/{epochs}] , loss: {total_loss/len(train_loader):.4f}')
  # Plot training loss
  print('Name: Sushmitha Gembunathan')
  print('Register Number:212224040342')
  plt.plot(train_losses, label='Training Loss')
  plt.xlabel('Epoch')
  plt.ylabel('MSE Loss')
  plt.title('Training Loss Over Epochs')
  plt.legend()
  plt.show()
train_model(model,train_loader,criterion,optimizer) 


```

### OUTPUT

## Training Loss Over Epochs Plot

<img width="669" height="713" alt="image" src="https://github.com/user-attachments/assets/8572150a-5db8-4e34-a8c2-c59761b6f045" />


## True Stock Price, Predicted Stock Price vs time

<img width="850" height="495" alt="image" src="https://github.com/user-attachments/assets/fb3b5180-4fa7-4fd8-af46-78a9884f0886" />


### Predictions
<img width="251" height="32" alt="image" src="https://github.com/user-attachments/assets/aa73fc08-42ed-40b6-9a8f-39ea4e5bc181" />


## RESULT
Historical stock price data was preprocessed and used to train an RNN model, which successfully minimized training loss and accurately predicted stock price trends as shown by the close alignment between actual and predicted values.

