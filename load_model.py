from keras.models import model_from_json
import cv2
import os
import numpy as np

# load the model:
with open('tiramisu_fc_dense67_model_12_func.json') as model_file:
    tiramisu = model_from_json(model_file.read())

tiramisu.load_weights('weights/prop_tiramisu_weights_67_12_func_10-e7_decay.best.hdf5')

val_data_path = "CamVid/val/0016E5_07959.png"
img = cv2.imread(val_data_path)
img = cv2.resize(img, dsize=(224,224))

input_data = np.array(img).reshape(-1,224,224,3)
prediction = tiramisu.predict(input_data)

print(prediction[0][0])
print(prediction.shape)
