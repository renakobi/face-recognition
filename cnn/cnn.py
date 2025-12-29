##################################################################################
#---------------------------------------------------------------------------------
#THIS IS THE FAILED CNN, ONLY HERE FOR RESEARCH REFERENCE CNN2 IS THE ONE BEING USED!!!!!!
#---------------------------------------------------------------------------------
##################################################################################













import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
#the parameters for cnn model
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 25 #epoch is the number of iterations of cnn kind of like layers but its to increase accuracy
LR = 1e-3 # lr is learning rate and its usually 1e - x where the lower the x the faster but riskier and viceversa

src = r"C:\Users\Hp\Desktop\uni\ML\project\Extracted Faces"
out = r"C:\Users\Hp\Desktop\uni\ML\project\data_split"

#We're using torchvision's imagefolder that expects directories and not arrays/ variables
train_dir = os.path.join(out, "train")
test_dir = os.path.join(out, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)#exist_ok to avoid error if directory already exists

all_images = []
all_labels = []

for person in os.listdir(src):
    person_path = os.path.join(src, person)
    if not os.path.isdir(person_path):
        continue
    for img in os.listdir(person_path):
        all_images.append(os.path.join(person_path, img))
        all_labels.append(person)

X_train, X_test, y_train, y_test = train_test_split(
    all_images, all_labels,
    test_size=0.2, random_state=42, stratify=all_labels
)

for img_path, label in zip(X_train, y_train):
    dest = os.path.join(train_dir, label)
    os.makedirs(dest, exist_ok=True)
    shutil.copy(img_path, dest)

for img_path, label in zip(X_test, y_test):
    dest = os.path.join(test_dir, label)
    os.makedirs(dest, exist_ok=True)
    shutil.copy(img_path, dest)

#prep and clean the data
train_trasnform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

train_ds = datasets.ImageFolder(train_dir, transform=train_trasnform)
test_ds  = datasets.ImageFolder(test_dir,  transform=test_transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

nb_classes = len(train_ds.classes)
print("Classes:", nb_classes)

class CNN(nn.Module):
    #why classes? PyToRcH requires containers and __init__ is the constructor where were just declaring the variables
    def __init__(self, num_classes):
        super().__init__()
        #c is convolution and that is the multiplication and summation of each kernel\ filter which is just a matrix ....the math
        # so conv2d(3 is input channels (rgb so 3 inputs for color), 32 is nb of kernels, 3 is size of matrix)
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        #basically taking input from conv2 and putting them into conv2 etc
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        #pooling is basically grouping a certain amount of pixels into bundles\pools so they are treated as the same value
        #this also helps with saving on computation costs (time\ resources etc...)
        self.pool = nn.MaxPool2d(2)
        #128 * 16 * 16 is inputs x height x width basically taking learned features as vectors to compute (classify them)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        #DROP OUT TO REDUCE OVERFITTING
        self.dropout = nn.Dropout(0.5)
        #256 aka 2^8 is the size to be compressed to and we compress in 2 steps to avoid overfitting basically were generalising the data with the first step
        self.fc2 = nn.Linear(256, num_classes)
        #fc1 reduces size and fc2 calculates score for each class(categorises)

    def forward(self, x):
        #here we're pooling each stage (conv1,2,3) and relu removes negatives
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        #flatten... reshapes all the dimensions of the array except the first which in this case is batch
        x = torch.flatten(x, 1)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)

model = CNN(nb_classes).to("cpu") #takes nb classes as nb of categories
criterion = nn.CrossEntropyLoss() # quantifies the models accuracy between 0 and 1. linguistically it means measuring the randomness
optimizer = optim.Adam(model.parameters(), lr=LR) #adam: Adaptive Moment Estimation, it updates weights on each step

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to("cpu"), y.to("cpu")
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1:02d} | train_loss={total_loss/len(train_loader):.4f}")

model.eval()
correct = 0
total = 0
top5_correct = 0

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to("cpu"), y.to("cpu")
        outputs = model(x)

        preds = outputs.argmax(1)
        correct += (preds == y).sum().item()

        top5 = outputs.topk(5, dim=1).indices
        top5_correct += (top5 == y.view(-1, 1)).any(dim=1).sum().item()

        total += y.size(0)

print("TOP-1 TEST ACCURACY:", correct / total)
print("TOP-5 TEST ACCURACY:", top5_correct / total)

torch.save(model.state_dict(), "cnn_tts_cpu.pth")
