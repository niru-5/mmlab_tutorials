import time
import os
import argparse
import json
from picamera2 import Picamera2, Preview


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json", help="Path to the config file")
    args = parser.parse_args()
    return args


def main(args):
    with open(args.config, "r") as f:
        config = json.load(f)
        
    image_folder = config["image_folder"]
    picam2 = Picamera2()
        
    # find all the sensor modes
    
    # and then for each sensor mode, take an image and save it
    
    modes = picam2.sensor_modes
    
    for i, mode in enumerate(modes):
        print (f" mode {i}: {mode}")
        picam2.sensor_mode = mode
        capture_config = picam2.create_still_configuration(raw={'format':mode['format'], 
                                                        'bit_depth':mode['bit_depth'], 
                                                        'fps':mode['fps']})
        picam2.configure(capture_config)
        picam2.start()
        
        time.sleep(1)
        # picam2.switch_mode_and_capture_file(capture_config, f"{image_folder}/image_{i}.jpeg")
        image = picam2.capture_image()
        image.save(f"{image_folder}/image_model_{i}.jpeg")
        picam2.stop()
        
    
        
    

if __name__ == "__main__":
    args = parse_args()
    main(args)