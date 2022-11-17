import numpy as np
import pandas as pd
import cv2
import tensorflow
from tensorflow.keras.models import load_model

model = load_model("model_new.h5")

haar = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def detect_face(img):
    coods = haar.detectMultiScale(img)
    return coods


def detection(img):
    predict_y = (model.predict(img.reshape(1, 224, 224, 3))
                 > 0.5).astype("int32")
    return predict_y[0][0]


def draw_label(img, text, pos, bg_color):
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, cv2.FILLED)

    end_x = pos[0]+text_size[0][0]+2
    end_y = pos[1]+text_size[0][1]-2

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, cv2.FILLED)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 0), 1, cv2.LINE_AA)


cap = cv2.VideoCapture(0)


while True:

    ret, frame = cap.read()
    img = cv2.resize(frame, (224, 224))
    y_pred = detection(img)

    coods = detect_face(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    for x, y, w, h in coods:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)

    if y_pred == 0:
        draw_label(frame, "Mask", (30, 30), (0, 255, 0))
    else:
        draw_label(frame, "Without Mask", (30, 30), (0, 0, 255))

    cv2.imshow('Window', frame)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
cv2.destroyAllWindows()
