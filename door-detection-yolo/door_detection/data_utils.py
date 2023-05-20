import cv2
import matplotlib.pyplot as plt



def show_img(img_path: str):
    
    img = cv2.imread(img_path)
    cv2.imshow("Image", img)
    cv2.waitKey(80000)
    cv2.destroyAllWindows()
    
def plot_img(img_path: str):
    
    img = cv2.imread(img_path)
    plt.imshow(img)
    