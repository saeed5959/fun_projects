import numpy as np
import cv2


from core.settings import Door_Yolo_CONFIG

label_path = Door_Yolo_CONFIG.label_path
config_path = Door_Yolo_CONFIG.config_path
conf = Door_Yolo_CONFIG.conf
thresh = Door_Yolo_CONFIG.thresh


def detection_yolo(img_path:str, model_path:str):

	with open(label_path) as file:
		LABELS = file.read().split("\n")
  
	# initialize a list of colors to represent each possible class label
	np.random.seed(42)
	COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")

	# load our YOLO model
	print("loading yolo model ...")
	net = cv2.dnn.readNetFromDarknet(config_path, model_path)


	image = cv2.imread(img_path)
	(H, W) = image.shape[:2]
 
	# determine only the *output* layer names that we need from YOLO
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
 
	# construct a blob from the input image and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes and
	# associated probabilities
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(ln)

	# initialize our lists of detected bounding boxes, confidences, and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []


	for output in layerOutputs:
		for detection in output:
			# extract the class ID and confidence
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			# filter out weak predictions
			if confidence > conf:
				# scale the bounding box coordinates back relative to the
				# size of the image, keeping in mind that YOLO actually
				# returns the center (x, y)-coordinates of the bounding
				# box followed by the boxes' width and height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
				# use the center (x, y)-coordinates to derive the top and
				# and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
    
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)


	# apply non-maxima suppression to suppress weak, overlapping bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf, thresh)


	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			if LABELS[classIDs[i]]=="door":
				# extract the bounding box coordinates
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])

	return x, y, w, h
    
    
