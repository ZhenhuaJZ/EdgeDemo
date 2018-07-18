from keras.models import model_from_json
import cv2
import os
import numpy as np

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
    val_data_path = "CamVid/val/0016E5_07959.png"
    img = cv2.imread(val_data_path)
    img = cv2.resize(img, dsize=(224,224))

    input_data = np.array(img).reshape(-1,224,224,3)

    return input_data


def main():
    tiramisu = load_model()
    load_weights(tiramisu)
    input_data = load_img()
    prediction = tiramisu.predict(input_data)

    print(prediction[0][0])
    print(prediction.shape)

if __name__ == '__main__':
    main()
