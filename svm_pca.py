import os
import cv2
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

IMG_SIZE = 128
SRC = r"C:\Users\Hp\Desktop\uni\ML\project\Extracted Faces"

images = []
labels = []
id_to_imgs = {}

for person in os.listdir(SRC):
    p = os.path.join(SRC, person)
    if not os.path.isdir(p):
        continue
    imgs = [os.path.join(p, f) for f in os.listdir(p)]
    imgs = [x for x in imgs if os.path.isfile(x)]
    if len(imgs) >= 20:
        id_to_imgs[person] = imgs

if len(id_to_imgs) < 37:
    raise ValueError(f"Only {len(id_to_imgs)} identities have >= 20 images, need 37.")

chosen_ids = random.sample(list(id_to_imgs.keys()), 37)

for person in chosen_ids:
    personPath = os.path.join(SRC, person)
    for imgName in os.listdir(personPath):
        imgPath = os.path.join(personPath, imgName)
        img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.equalizeHist(img)
        images.append(img.flatten())
        labels.append(person)

X = np.array(images)
y = np.array(labels)

le = LabelEncoder()
yEncoded = le.fit_transform(y)

XTrain, XTest, yTrain, yTest = train_test_split(
    X, yEncoded, test_size=0.2, stratify=yEncoded, random_state=42
)

print("Num classes:", len(np.unique(yEncoded)))

best_acc = 0
best_params = None
best_pca = None
best_svm = None

for nComp in [50, 100, 150, 200, 300, 400]:
    pca = PCA(n_components=nComp, whiten=True, random_state=42)
    Xtr = pca.fit_transform(XTrain)
    Xte = pca.transform(XTest)

    for C in [0.1, 1, 10, 50, 100]:
        for gamma in ["scale", 1e-4, 1e-3, 3e-3, 1e-2]:
            svm = SVC(kernel="rbf", C=C, gamma=gamma)
            svm.fit(Xtr, yTrain)
            preds = svm.predict(Xte)
            acc = accuracy_score(yTest, preds)

            if acc > best_acc:
                best_acc = acc
                best_params = (nComp, C, gamma)
                best_pca = pca
                best_svm = svm
                print("NEW BEST:", best_acc, best_params)

print("BEST FINAL ACCURACY:", best_acc)
print("BEST PARAMS (nComp, C, gamma):", best_params)
print("EXPLAINED VARIANCE:", best_pca.explained_variance_ratio_.sum())

trainPCA = best_pca.transform(XTrain)

def showMatchesPCA(imagePath, topK=5):
    img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Invalid image path")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.equalizeHist(img)
    queryVec = img.flatten().reshape(1, -1)
    queryPCA = best_pca.transform(queryVec)

    dists = np.linalg.norm(trainPCA - queryPCA, axis=1)
    topIdx = np.argsort(dists)[:topK]

    plt.figure(figsize=(3 * (topK + 1), 4))

    plt.subplot(1, topK + 1, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Query")
    plt.axis("off")

    for i, idx in enumerate(topIdx, start=2):
        matchImg = XTrain[idx].reshape(IMG_SIZE, IMG_SIZE)
        label = le.inverse_transform([yTrain[idx]])[0]
        dist = dists[idx]

        plt.subplot(1, topK + 1, i)
        plt.imshow(matchImg, cmap="gray")
        plt.title(f"{label}\n{dist:.2f}")
        plt.axis("off")

    plt.show()
testImage = r"C:\Users\Hp\Desktop\uni\ML\project\0.jpg"
showMatchesPCA(testImage, topK=5)
