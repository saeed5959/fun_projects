import argparse

from door_detection.door_main import detection

def door_detection(img_path: str, model_path: str):
    
    img_out_path = detection(img_path, model_path)
    
    print(f"output image with door detection , has been saved in {img_out_path}")
    
    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, required=False)
    parser.add_argument("--model_path", type=str, required=False)
    args = parser.parse_args()
    

    door_detection(args.img_path, args.model_path)
        
                
