# reference for haarcade files
# https://github.com/opencv/opencv/tree/master/data/haarcascades
import cv2

video = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
haarsmile = cv2.CascadeClassifier('haarcascade_smile.xml')
smiled = []
faced = []
facecount = 0

# video_copy = video.copy()

while True:

    check, frame = video.read()

    video_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    thresh, binary = cv2.threshold(video_gray, 50, 255, cv2.THRESH_BINARY)

    facesb = haarsmile.detectMultiScale(binary, 1.3, 5)
    smiles = haarsmile.detectMultiScale(binary, 1.2, 5)

    # print(str(len(facesb))+" people faces detected")
    # print(str(len(smiles))+" people smiles detected")
    facecount = facecount + len(facesb)
    print(facecount)

    # contours, hierarchy = cv2.findContours(
    #    binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # detect faces and only smiles in faces
    for faceb in facesb:
        for smile in smiles:
            fxb, fyb, fwb, fhb = faceb
            sxb, syb, swb, shb = smile
            faced = cv2.rectangle(
                frame, (fxb, fyb), ((fxb+fwb), (fyb+fhb)), (0, 255, 0), 5)
            smiled = cv2.rectangle(
                faced, (sxb, syb), ((sxb+swb), (syb+shb)), (0, 255, 0), 5)

    cv2.imshow("image show", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
