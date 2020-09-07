import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array

model = model_from_json(open("model.json", "r").read())
model.load_weights('model.h5')
face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
emo_detect = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
cap = cv2.VideoCapture(0)

while True:
    ret, test_image = cap.read()
    if ret is False:
        print('Unable to load frame')
        break

    gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

    faces_detected = face_haar_cascade.detectMultiScale(gray, 1.05, 6)
    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_image, (x,y), (x+w, y+h), (255, 0, 0))
        roi_gray = gray[y:y+w, x:x+h]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        image_pixels = img_to_array(roi_gray)
        image_pixels = np.expand_dims(image_pixels, axis = 0)
        #image_pixels /= 255

        prediction = model.predict(image_pixels)
        max_index = np.argmax(prediction)

        emo_predict = emo_detect[max_index]

        cv2.putText(test_image, emo_predict, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        #cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 3)

        #resize_image = cv2.resize(test_image, (1000, 700))
        cv2.imshow('Emotion', test_image)
        if cv2.waitKey(10) == ord('x'):
            break

    cap.release()
    cv2.destroyAllWindows()

