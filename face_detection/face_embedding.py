import cv2
import sys
import face_recognition
import json


def Face_Embed(name,path):
    
    img = cv2.imread(path)
    rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    box = face_recognition.face_locations(rgb,model='hog')
    # compute the facial embedding for the any face
    encoding = face_recognition.face_encodings(rgb, box)[0]
    
    try:# open database if it exists
        with open("database.json", mode="r") as file:
            data = json.load(file)
    except:# make a new database if there is not
        with open("database.json", mode="w") as file:
            data = []
            json.dump(data,file)


    with open("database.json", mode="w") as file:
        # find in database if there was a name similar to new name
            duplicate_index = [i for (i,x) in enumerate(data) if x["name"]==name]
            # write in database name and encoder vector
            if duplicate_index==[]:
                data.append({"name":name , "encod":encoding.tolist()})
                json.dump(data, file)
            else:
                data[int(duplicate_index[0])] = {"name":name , "encod":encoding.tolist()}
                json.dump(data, file)


# Face_Embed('ali','/home/saeed/me.jpg')
if __name__ == '__main__':
    Face_Embed(sys.argv[1] , sys.argv[2])