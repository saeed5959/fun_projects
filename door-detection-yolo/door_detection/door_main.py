from door_detection.door_yolo import detection_yolo
from door_detection.door_cv2 import detection_cv2


def detection(img_path: str, model_path: str):
    
    #detect approximate boundary of door
    x, y, w, h = detection_yolo(img_path, model_path)
    
    #detect exact boindary of door based on detection_yolo
    img_out_path = detection_cv2(img_path, x, y, w, h)
    
    
    
    return img_out_path
