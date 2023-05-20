import cv2
import sys
import face_recognition
import json
import numpy as np


def Face_Compare(name,path):
    
    img = cv2.imread(path)
    rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    box = face_recognition.face_locations(rgb,model='hog')
    # compute the facial embedding for the any face
    new_encoding = face_recognition.face_encodings(rgb, box)[0]
    # open database
    with open("database.json", mode="r") as file:
        data = json.load(file)

    # find the reference image encoder based on name
    old_encoding = np.array([x["encod"] for x in data if x["name"]==name])
    if old_encoding == []:
        return print("you hav not uploaded reference image")
    else:
        comparision = face_recognition.compare_faces([old_encoding],new_encoding,tolerance=0.07)[0]
        similarity = np.where(comparision==True)[0].size / comparision.size
        print(similarity)
        if similarity > 0.8:
            return print("verification is correct")
        else:
            return print("verifiation is not correct")



# Face_Compare('ali','/home/saeed/s.png')
if __name__ == '__main__':
    Face_Compare(sys.argv[1] , sys.argv[2])