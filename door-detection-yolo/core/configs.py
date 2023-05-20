

class DoorYolo:
    """
        RBC counting config
    """

    def __init__(self):
        self.label_path: str = "./configs/coco.names"
        self.config_path: str = "./configs/yolov3.cfg"
        self.conf: int = 0.5
        self.thresh:int = 0.3

        

class DoorCv2:
    """
        WBC classify config
    """

    def __init__(self):        
        self.scale:int = 150
        self.sigma:int = 0.5
        self.min_size:int = 5000

        
  