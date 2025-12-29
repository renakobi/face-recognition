import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

IMG_SIZE = 128
DEVICE = "cpu"

demoTransform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

src = r"C:\Users\Hp\Desktop\uni\ML\project\Extracted Faces"
modelPath = "softmax_then_triplet_cpu.pth"

# same model architecture as training
class FaceNetScratch(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.drop = nn.Dropout(0.5)

        self.embed = nn.Linear(256, emb_dim)

    def features(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.drop(torch.relu(self.fc1(x)))
        return x

    def forwardEmbed(self, x):
        e = self.embed(self.features(x))
        return torch.nn.functional.normalize(e, p=2, dim=1)

# build gallery (embeddings + images)
galleryEmb = []
galleryLab = []
galleryImg = []

labelMap = {}
currentLabel = 0

model = FaceNetScratch(emb_dim=128).to(DEVICE)
state = torch.load(modelPath, map_location=DEVICE)
model.load_state_dict(state, strict=False)
model.eval()


with torch.no_grad():
    for person in os.listdir(src):
        personPath = os.path.join(src, person)
        if not os.path.isdir(personPath):
            continue

        if person not in labelMap:
            labelMap[person] = currentLabel
            currentLabel += 1

        for imgName in os.listdir(personPath):
            imgPath = os.path.join(personPath, imgName)
            img = Image.open(imgPath).convert("RGB")
            imgTensor = demoTransform(img).unsqueeze(0).to(DEVICE)

            emb = model.forwardEmbed(imgTensor).cpu()
            galleryEmb.append(emb)
            galleryLab.append(labelMap[person])
            galleryImg.append(img)

galleryEmb = torch.cat(galleryEmb)
galleryLab = torch.tensor(galleryLab)

idToName = {v: k for k, v in labelMap.items()}

# visual identification
def showMatches(imagePath, topK=5):
    img = Image.open(imagePath).convert("RGB")
    imgTensor = demoTransform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        queryEmb = model.forwardEmbed(imgTensor).cpu()

    sims = queryEmb @ galleryEmb.T
    topIdx = sims.topk(topK, dim=1).indices[0]
    topScores = sims[0, topIdx]

    plt.figure(figsize=(3 * (topK + 1), 4))

    plt.subplot(1, topK + 1, 1)
    plt.imshow(img)
    plt.title("Query")
    plt.axis("off")

    for i, (idx, score) in enumerate(zip(topIdx, topScores), start=2):
        matchImg = galleryImg[idx]
        label = idToName[galleryLab[idx].item()]

        plt.subplot(1, topK + 1, i)
        plt.imshow(matchImg)
        plt.title(f"{label}\n{score:.2f}")
        plt.axis("off")

    plt.show()

# run demo
testImage = r"C:\Users\Hp\Desktop\uni\ML\project\0.jpg"
showMatches(testImage, topK=5)
