import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"---------------------device:{device}------------------------------------")
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))

])
#load train and testdata
traindata=torchvision.datasets.MNIST(root='./data',download=True,train=True,transform=transform)
loadtrain=torch.utils.data.DataLoader(traindata,batch_size=64,shuffle=True)
testdata=torchvision.datasets.MNIST(root='./data',download=True,train=False,transform=transform)
loadtest=torch.utils.data.DataLoader(testdata,shuffle=False,batch_size=64)

print("--------------LOADED SUCCESSFULLY---------------------")
#CONVOLUTION,PADDING,DROPOUT(REGULARIZATION),FLATTENING
class digit(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,32,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.dropout=nn.Dropout(0.25)
        self.fc1=nn.Linear(64*7*7,128)
        self.fc2=nn.Linear(128,10)
    def forward(self,x):
        x=self.pool(torch.relu(self.conv1(x)))
        x=self.pool(torch.relu(self.conv2(x)))
        x=x.view(-1,7*7*64)
        x=torch.relu(self.fc1(x))
        x=self.dropout(x)
        x=self.fc2(x)
        return x
model=digit().to(device)
#loss and optimization
critaria=nn.CrossEntropyLoss()
optimization=optim.Adam(model.parameters(),lr=0.001)

#TRAINING YIPEEEE
print("Training phase...............")
model.train()
for epoch in range(1):
    running_loss=0.0
    for i,data in enumerate(loadtrain,0):
        inputs,labels=data
        inputs=inputs.to(device)
        labels=labels.to(device)
        optimization.zero_grad()
        output=model(inputs)
        #input,backward,optimization
        loss=critaria(output,labels)
        loss.backward()
        optimization.step()
        running_loss+=loss.item()
        if i%100==99:
            print(f"batch:{i+1},loss:{running_loss/100:.4f}")
            running_loss=0.0
print("Training is finished.................................")


#TESTING EVALUATION STUFF
model.eval()
dataiter=iter(loadtest)
images,labels=next(dataiter)
#for 1st image
testimg=images[10].unsqueeze(0).to(device)
output=model(testimg)
_,prediction=torch.max(output,1)
print(f"ACTUALOUTPUT:{labels[10].item()}")
print(f"preidiction:{prediction.item()}")
