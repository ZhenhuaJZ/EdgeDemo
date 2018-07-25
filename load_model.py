from keras.models import model_from_json
import cv2
import os
import numpy as np
from helper import *

model_name = 'tiramisu_fc_dense67_model_12_func.json'
weights_path = 'weights/prop_tiramisu_weights_67_12_func_10-e7_decay.best.hdf5'
# load the model:
def load_model():

    with open(model_name) as model_file:
        tiramisu = model_from_json(model_file.read())

    return tiramisu

def load_weights(tiramisu):

    tiramisu.load_weights(weights_path)

def load_img():

    val_data_path = "CamVid/val/0016E5_08029.png"
    img = np.rollaxis(normalized(cv2.imread(val_data_path)[136:,256:]),2)
    input_data = np.array(img).reshape(-1,224,224,3)

    return input_data

def color_map(result):
    # FYI they are:
    # Sky = [128,128,128]
    # Building = [128,0,0]
    # Pole = [192,192,128]
    # Road_marking = [255,69,0]
    # Road = [128,64,128]
    # Pavement = [60,40,222]
    # Tree = [128,128,0]
    # SignSymbol = [192,128,128]
    # Fence = [64,64,128]
    # Car = [64,0,128]
    # Pedestrian = [64,64,0]
    # Bicyclist = [0,128,192]
    # Unlabelled = [0,0,0]
    color_coding = [[128,128,128],[128,0,0],[192,192,128],
                    [255,69,0],[128,64,128],[60,40,222],
                    [128,128,0],[192,128,128],[64,64,128],
                    [64,0,128],[64,64,0],[0,128,192]]

    color_map = np.zeros((224,224,3), np.uint8)

    for i in range(result.shape[0]):
        for d in range(result.shape[1]):
            color_map[i,d] = color_coding[result[i,d]]
    return color_map

def main():
    tiramisu = load_model()
    print("model_loaded")
    load_weights(tiramisu)
    print("weights_loaded")
    input_data = load_img()
    print("img_loaded")
    prediction = tiramisu.predict(input_data)
    prediction = tiramisu.predict(input_data).reshape(224,224,12)
    # get color matrix
    result = [np.argmax(prediction[i], axis=1) for i in range(224)]
    result = np.array(result)

    map = color_map(result)

    cv2.imshow("color_map", map)
    cv2.resizeWindow('color_map', 600,600)
    cv2.waitKey(50000)

if __name__ == '__main__':
    main()
