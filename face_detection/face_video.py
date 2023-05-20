import cv2
import sys

def face_detect_video(path):
    
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Read the input image
    video = cv2.VideoCapture(path)
    while True:
        _,img = video.read()
        # Convert into grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        # Draw rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Display the output
        cv2.namedWindow('a', cv2.WINDOW_NORMAL)
        cv2.imshow('a',img)
        cv2.waitKey(2)

if __name__ == '__main__':
    face_detect_video(sys.argv[1])
# face_detect_video("/home/saeed/me.mp4")