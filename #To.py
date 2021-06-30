# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import cv2 as cv


# %%
def rescaler(img, scale=0.5):
    w, h = int(img.shape[1] * scale), int(img.shape[0] * scale)
    dim = (w , h)
    return cv.resize(img, dim, interpolation=cv.INTER_AREA)


def changeResolution(capture, width, height):
    capture.set(3, width)
    capture.set(4, height)


# %%
live = cv.VideoCapture(0)
live.set(3, 1250)
live.set(4, 750)

while True:
    isTrue, frame = live.read()
    cv.imshow("live Video", frame)

    if cv.waitKey(20) & 0xFF==ord('q'):
        break

live.release()
cv.destroyAllWindows()


# %%
import cv2 as cv
import numpy as np

image = cv.imread("./index.jpeg")


warp_affine = cv.warpAffine(image, np.float32([[1,0, 90], [0,1, 20]]), (image.shape[1], image.shape[0]))
flipped = cv.flip(image, 1)
resize = cv.resize(image, (900, 700), interpolation=cv.INTER_LINEAR)

cv.imshow("Original Image",image)
cv.imshow("Resized Image",resize)
cv.imshow("shifted Image",warp_affine)
cv.imshow("Flipped Image",flipped)
cv.waitKey()
cv.destroyAllWindows()


# %%
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

image  = cv.imread("./peets.jpg")
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
blank = np.zeros(gray.shape[:2], dtype="uint8")

circle = cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 200, 255, -1, 2)
mask = cv.bitwise_and(gray, gray, mask= circle)
# cv.imshow("Mask image", mask)
hist = cv.calcHist([gray], [0], mask, [456], [0, 256])

colors = ["b", "g", "r"]

for i, c in enumerate(colors):
    histo = cv.calcHist([image], [i], None, [255], [0, 256])
    plt.plot(histo, color=c)
    plt.xlim([0, 256])

# plt.plot(hist)
plt.title("histogram of the image")
plt.xlabel("bins")
plt.ylabel("intensity")
# plt.tight_layout()
plt.show()

# cv.imshow("Original Image", image)
cv.waitKey()
cv.destroyAllWindows()


# %%



# %%



# %%



# %%
image = cv.imread("./index.jpeg")

gray =  cv.cvtColor(image, cv.COLOR_BGR2GRAY)

blur = cv.GaussianBlur(gray, (3,3), cv.BORDER_DEFAULT)

edge = cv.Canny(blur, 180 ,170)

dilated = cv.dilate(edge, (7, 9), iterations=4)
eroded = cv.erode(dilated, (7, 2), iterations=1)

cv.imshow("Original Image",image)
cv.imshow("Gray Image",gray)
cv.imshow("BLur Image",blur)
cv.imshow("edges Image",edge)
cv.imshow("dilate Image", dilated)
cv.imshow("eroded Image", eroded)
cv.waitKey()
cv.destroyAllWindows()

# %% [markdown]
# # Threshold party

# %%
import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt

image = cv.imread("./cats_2.jpg")
image = cv.resize(image, (500, 500))
gray = cv.cvtColor(image , cv.COLOR_BGR2GRAY)

threshold, thresh_im = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
threshold, thresh_im_inv  = cv.threshold(thresh_im, 150, 255, cv.THRESH_BINARY_INV)

# Adaptive Threshold 

adaptive = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 7, 2)

cv.imshow("Threshold Image ",thresh_im_inv)
cv.imshow("Adaptive thresh ",adaptive)
cv.waitKey()
cv.destroyAllWindows() 

# %% [markdown]
# # Edges detector and drawer

# %%
import cv2 as cv
import numpy as np

image = cv.imread("./pet.jpg")
image = cv.resize(image, (350, 300))
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow("Gray Image", gray)

lab = cv.Laplacian(gray, cv.CV_16S)
lab =  np.uint8(np.absolute(lab))


sobelx = cv.Sobel(gray, cv.CV_64F, 0, 1)
sobely = cv.Sobel(gray, cv.CV_64F, 1, 0)

canny = cv.Canny(gray, 120, 150)

cv.imshow("laplacian image", lab)
cv.imshow("Sobel X image", sobelx)
cv.imshow("Sobel Y image", sobely)
cv.imshow("Canny Image", canny)
cv.waitKey()
cv.destroyAllWindows()


# %%
import cv2 as cv

HAAR_FACE = cv.CascadeClassifier("./haar_faces.xml")
chris = cv.imread("./chris2.jpg")
chris = cv.resize(chris, (900, 700))
gray = cv.cvtColor(chris, cv.COLOR_BGR2GRAY)
# cv.imshow("Person Original Image", chris)
# cv.imshow("Person Gray Image", gray)

faces_rect = HAAR_FACE.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors=2)

for (x,y, w, h) in faces_rect:
    cv.rectangle(chris, (x, y), (x+w, y+h), (29, 90, 250), thickness = 2)


cv.imshow("Detected Faces", chris)

print(len(faces_rect))
cv.waitKey()
cv.destroyAllWindows()


# %%
import cv2 as cv

live = cv.VideoCapture(0)
live.set(3, 950)
live.set(4, 750)
HAAR_FACE = cv.CascadeClassifier("./data/haarcascade_frontalface_alt2.xml")


while True:
    isTrue, frame = live.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frames = HAAR_FACE.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors=2)
    for (x,y, w, h) in frames:
        cv.rectangle(frame, (x, y), (x+w, y+h), (29, 90, 250), thickness = 2)
        roi = gray[y:h+y, x:x+w]
        roi_c = frame[y:h+y, x:x+w]
        cv.imwrite("./chris_face.png", roi)
        cv.imwrite("./chris_face_colored.png", roi_c)

    cv.imshow("live Video detected faces ", frame)
    if cv.waitKey(20) & 0xFF==ord('q'):
        break

live.release()
cv.destroyAllWindows()


# %%
import os
import cv2 as cv
import numpy as np

path = "./persons"

X_train = []
y_train = []
HAAR_FACE = cv.CascadeClassifier("./data/haarcascade_frontalface_alt2.xml")


for root, dirs, files in os.walk(path):
    
    for file in files:
        path = os.path.join(root, file)
        label = os.path.basename(root)
        image = cv.imread(path)
        image_array = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        faces = HAAR_FACE.detectMultiScale(image_array, scaleFactor = 1.1, minNeighbors=2)
        
        for x, y, w,h in faces: 
            cv.rectangle(faces, (x, y), (x+w, y+h), (29, 90, 250), thickness = 2)
            roi = faces[y:h+y, x:x+w]
            X_train.append(roi)
            y_train.append(label)




# %%
print(len(X_train))
features = np.array(X_train, dtype="object")
labels = np.array(y_train)
print(type(labels))
print(type(features))
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features, labels)

face_recognizer.save("recognizer.yaml")

np.save("features.npy", X_train)
np.save("label.npy", y_train)


# %%



