import cv2 as cv


live = cv.VideoCapture(0)
live.set(3, 850)
live.set(4, 750)

HAAR_FACE = cv.CascadeClassifier("./data/haarcascade_frontalface_alt2.xml")
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read("recognizer.yml")


peoples= ['TonyStark', 'chrisKid', 'elonMusk']


while True:
    isTrue, frame = live.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frames = HAAR_FACE.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors=3)
    for (x,y, w, h) in frames:

        roi = gray[y:h+y, x:x+w]
        roi_c = frame[y:y+h, x:x+w]

        label, confidence = face_recognizer.predict(roi)
        text = f"{peoples[label]}({confidence :.2f}%)"

        cv.rectangle(frame, (x, y), (x+w, y+h), (29, 90, 250), thickness = 2)
        cv.putText(frame, text, (x+2,y-4), cv.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv.LINE_AA)
        
        cv.imwrite("./chris_face.png", roi)
        cv.imwrite("./chris_face_colored.png", roi_c)

    cv.imshow("live Video detected faces ", frame)
    if cv.waitKey(20) & 0xFF==ord('q'):
        break

live.release()
cv.destroyAllWindows()