import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

#TRACING:
#
#The script first filters the dataset to keep only identities with
#enough images and splits the images into training and testing sets.

#Next, a CNN is trained using softmax classification so
#the network learns basic identity-discriminative features.

#After this initial structure is learned,
#the model is fine-tuned using triplet loss to reshape
#the feature space for similarity-based recognition.

#Once training is complete, face embeddings are 
#extracted for all images and evaluated using
#k-nearest neighbor retrieval, Recall@K, and verification metrics.

#Finally, the trained model is saved for later use and visual demos.
#
IMG_SIZE = 128
EPOCHS_SOFTMAX = 15
BATCH_SOFTMAX = 32
LR_SOFTMAX = 1e-3

EPOCHS_TRIPLET = 25
LR_TRIPLET = 3e-4
P = 8
K = 4
src = r"C:\Users\Hp\Desktop\uni\ML\project\Extracted Faces"

random.seed(42)
torch.manual_seed(42)

class FaceDataset(Dataset):
#Loads face images from disk, applies transforms, 
#and returns image+label pairs so PyTorch can feed data to the model
#during train and eval.
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

class BalancedBatchSampler(Sampler):
#Builds training batches that contain multiple images per identity,
#which is required for triplet loss to form valid anchor|positive|negative pairs.
    def __init__(self, labels, p, k, seed=42):
        self.labels = labels
        self.p = p
        self.k = k
        self.rng = random.Random(seed)

        self.label_to_indices = {}
        for idx, lab in enumerate(labels):
            self.label_to_indices.setdefault(lab, []).append(idx)

        self.valid_labels = [lab for lab, idxs in self.label_to_indices.items() if len(idxs) >= k]
        self.num_batches = len(labels) // (p * k)

    def __iter__(self):
        for _ in range(self.num_batches):
            chosen = self.rng.sample(self.valid_labels, self.p)
            batch = []
            for lab in chosen:
                idxs = self.label_to_indices[lab]
                batch.extend(self.rng.sample(idxs, self.k))
            yield batch

    def __len__(self):
        return self.num_batches


train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

test_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

id_to_imgs = {}
for person in os.listdir(src):
    p = os.path.join(src, person)
    if not os.path.isdir(p):
        continue
    imgs = [os.path.join(p, f) for f in os.listdir(p)]
    imgs = [x for x in imgs if os.path.isfile(x)]
    if len(imgs) >= 20:
        id_to_imgs[person] = imgs


chosen_ids = random.sample(list(id_to_imgs.keys()), 37)
label_map = {pid: i for i, pid in enumerate(chosen_ids)}

paths, labels = [], []
for pid in chosen_ids:
    for img in id_to_imgs[pid]:
        paths.append(img)
        labels.append(label_map[pid])

X_tr, X_te, y_tr, y_te = train_test_split(
    paths, labels, test_size=0.2, stratify=labels, random_state=42
)

num_classes = len(set(y_tr))
print("Classes:", num_classes)

train_ds = FaceDataset(X_tr, y_tr, transform=train_tf)
test_ds  = FaceDataset(X_te, y_te, transform=test_tf)

train_loader_softmax = DataLoader(train_ds, batch_size=BATCH_SOFTMAX, shuffle=True, num_workers=0)
test_loader_eval     = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
train_loader_eval    = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=0)

train_loader_triplet = DataLoader(
    train_ds,
    batch_sampler=BalancedBatchSampler(y_tr, P, K, seed=42),
    num_workers=0
)

class FaceNetScratch(nn.Module):
#Defines the cnn that extracts facial features,
#produces embeddings for similarity comparison, 
#and outputs class scores during softmax training.
    def __init__(self, num_classes, emb_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.drop = nn.Dropout(0.5)

        self.classifier = nn.Linear(256, num_classes)
        self.embed = nn.Linear(256, emb_dim)

    def features(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.drop(torch.relu(self.fc1(x)))
        return x

    def forward_softmax(self, x):
        f = self.features(x)
        return self.classifier(f)

    def forward_embed(self, x):
        f = self.features(x)
        e = self.embed(f)
        return torch.nn.functional.normalize(e, p=2, dim=1)

model = FaceNetScratch(num_classes=num_classes, emb_dim=128).to("cpu")

criterion = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=LR_SOFTMAX)

for epoch in range(EPOCHS_SOFTMAX):
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in train_loader_softmax:
        x = x.to("cpu")
        y = y.to("cpu")

        opt.zero_grad()
        logits = model.forward_softmax(x)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()

        total_loss += loss.item()
        n += 1

    print(f"SOFTMAX Epoch {epoch+1:02d} | train_loss={total_loss/max(1,n):.4f}")


opt2 = optim.Adam(model.parameters(), lr=LR_TRIPLET)

def batch_hard_triplet_loss(emb, lab, margin):
    d = torch.cdist(emb, emb, p=2)
    losses = []

    for i in range(emb.size(0)):
        pos_mask = (lab == lab[i])
        neg_mask = (lab != lab[i])
        pos_mask[i] = False

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue

        hardest_pos = d[i][pos_mask].max()
        hardest_neg = d[i][neg_mask].min()
        losses.append(torch.relu(hardest_pos - hardest_neg + margin))

    if len(losses) == 0:
        return torch.tensor(0.0, device=emb.device)
    return torch.stack(losses).mean()

for epoch in range(EPOCHS_TRIPLET):
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in train_loader_triplet:
        x = x.to("cpu")
        y = y.to("cpu")

        opt2.zero_grad()
        emb = model.forward_embed(x)
        loss = batch_hard_triplet_loss(emb, y, 0.2)
        loss.backward()
        opt2.step()

        total_loss += loss.item()
        n += 1

    print(f"TRIPLET Epoch {epoch+1:02d} | train_loss={total_loss/max(1,n):.4f}")

model.eval()

def extract_embeddings(loader):
    embs = []
    labs = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to("cpu")
            e = model.forward_embed(x).cpu()
            embs.append(e)
            labs.append(y)
    return torch.cat(embs), torch.cat(labs)

train_emb, train_lab = extract_embeddings(train_loader_eval)
test_emb,  test_lab  = extract_embeddings(test_loader_eval)

def knn_topk(train_emb, train_lab, test_emb, test_lab, k=5):
    sims = test_emb @ train_emb.T
    topk_idx = sims.topk(k, dim=1).indices
    topk_labels = train_lab[topk_idx]
    top1 = (topk_labels[:, 0] == test_lab).float().mean().item()
    topk_acc = (topk_labels == test_lab.view(-1, 1)).any(dim=1).float().mean().item()
    return top1, topk_acc

def recall_at_k(train_emb, train_lab, test_emb, test_lab, k):
    _, r = knn_topk(train_emb, train_lab, test_emb, test_lab, k=k)
    return r

def verification_metrics(emb, lab):
    scores = []
    targets = []
    n = emb.size(0)
    for i in range(n):
        for j in range(i + 1, n):
            sim = torch.dot(emb[i], emb[j]).item()  #cosine similarity
            same = int(lab[i] == lab[j])
            scores.append(sim)
            targets.append(same)
    scores = np.array(scores, dtype=np.float32)
    targets = np.array(targets, dtype=np.int32)

    th = np.median(scores)
    preds = (scores >= th).astype(np.int32)
    acc = (preds == targets).mean()

    auc = roc_auc_score(targets, scores) if len(np.unique(targets)) > 1 else float("nan")
    return acc, auc

top1, top5 = knn_topk(train_emb, train_lab, test_emb, test_lab, k=5)
print("kNN top 5:", top5)

print("Recall@1:", recall_at_k(train_emb, train_lab, test_emb, test_lab, 1))
print("Recall@5:", recall_at_k(train_emb, train_lab, test_emb, test_lab, 5))
print("Recall@10:", recall_at_k(train_emb, train_lab, test_emb, test_lab, 10))

ver_acc, auc = verification_metrics(test_emb, test_lab)
print("Verification accuracy:", ver_acc)
print("ROC AUC:", auc)

torch.save(model.state_dict(), "CNNtrained.pth")

