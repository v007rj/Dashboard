#Import Statements----------------------------------------------
###I will update this#######3##############
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam

cla = Sequential()
cla.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation='relu'))
cla.add(MaxPooling2D(2, 2))
cla.add(Dropout(0.3))
cla.add(Conv2D(32, (3,3), activation='relu'))
cla.add(MaxPooling2D(2,2))
cla.add(Flatten())
cla.add(Dense(units=128, activation='linear'))
cla.add(Dense(units=3, activation='softmax'))

opt = Adam(lr = 1e-6)
cla.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('cnn/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32)

test_set = test_datagen.flow_from_directory('cnn/test',
                                            target_size = (64, 64),
                                            batch_size = 32)

cla.fit_generator(training_set,
                 steps_per_epoch = 1460,
                 epochs = 1,
                 validation_data = test_set,
                 validation_steps = 76)

import numpy as np
from keras.preprocessing import image

test_image = image.load_img('person_045.bmp', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
cla.predict(test_image)

import numpy as np
import cv2
cap = cv2.VideoCapture(0)
while(True):
    res, pic = cap.read()
    cv2.imwrite('pic.jpg', pic)
    
    test_image = image.load_img('pic.jpg', target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    res = cla.predict(test_image)
    
    if np.argmax(res[0] == 0):
        cv2.putText(pic,('Crunches - '+ str(res[0][0])),
                    (320,361),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255,255,255),2)
        yo = cv2.imread('crunches1.jpg')
        cv2.imshow('crunches', yo)
    if np.argmax(res[0] == 1):
        cv2.putText(pic,'Relaxed',
                    (320,361),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255,255,255),2)
    if np.argmax(res[0] == 2):
        cv2.putText(pic,'Squat',
                    (320,361),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255,255,255),2)    
    
    cv2.imshow('pic', pic) 
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()





