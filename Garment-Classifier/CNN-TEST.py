from keras import backend as k
from keras.models import model_from_json
import cv2
import numpy as np

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")

labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

img = cv2.imread('C:/Users/RAWAT/Desktop/tee.jpg', cv2.IMREAD_GRAYSCALE)
# im_resize = cv2.resize(img, (28, 28))
# cv2.imshow("img", im_resize)
# cv2.waitKey(0)
# print(labels[np.argmax(prediction)])

# for i in range(5):
#     plt.grid(False)
#     plt.imshow(test_img[i], cmap = plt.cm.binary)
#     plt.xlabel("Actual: "+ labels[test_labels[i]])
#     plt.title("Prediction "+ labels[np.argmax(prediction[i])])
#     plt.show()