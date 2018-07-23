from keras.models import model_from_json
import cv2
import os
import numpy as np

model_name = 'tiramisu_fc_dense67_model_12_func.json'
weights_path = 'weights/prop_tiramisu_weights_67_12_func_10-e7_decay150.hdf5'
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

def color_map(result):

    color_coding = [[255,0,0],[0,255,0],[0,0,255],
                    [255,255,0],[255,0,255],[0,255,255],
                    [125,0,0],[0,125,0],[0,0,125],
                    [125,125,0],[125,0,125],[0,125,125]]

    color_map = np.zeros((224,224,3), np.uint8)
    # cv2.imshow("color_map", color_map)
    # cv2.waitKey(1000000)

    print(result.shape[0])
    print(result.shape[1])
    print(result[0])
    # print(result[1,3])
    for i in range(result.shape[0]):
        for d in range(result.shape[1]):
            color_map[i,d] = color_coding[result[i,d]]
            print("[{},{}]".format(i,d))
            print("result : ", result[i,d])
            # print(result[i,d])
        # exit()
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
    print(result)

    map = color_map(result)

    cv2.imshow("color_map", map)
    cv2.resizeWindow('color_map', 600,600)
    cv2.waitKey(1000000)
    #print(prediction)
    # print(prediction.shape)
    # w, h, c = np.indices(prediction.shape)
    # print(c.shape)
    # result = np.argmax(prediction, axis=1)
    # print(result[0])
    # print(result.shape)
    #np.argmax()



if __name__ == '__main__':
    main()
