from PyQt5 import QtCore, QtGui, QtWidgets 
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QFileDialog
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import cv2

TF_MODEL_URL = 'https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1'
IMAGE_SHAPE = (321 , 321)

data = pd.read_csv('landmarks_classifier_asia_V1_label_map.csv')

classifier = tf.keras.Sequential([hub.KerasLayer(
    TF_MODEL_URL,
    input_shape = IMAGE_SHAPE + (3,),
    output_key = "predictions:logits"
)])

label_map = dict(zip(data.id , data.name))

def classifyimg(RGBimg):
    RGBimg = np.array(RGBimg) / 255
    RGBimg = np.reshape(RGBimg , (1,321 , 321 , 3))
    predictions = classifier.predict(RGBimg)
    return label_map[np.argmax(predictions)]

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(303, 472)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(-60, -50, 391, 561))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.graphicsView.setFont(font)
        self.graphicsView.setAutoFillBackground(False)
        self.graphicsView.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.graphicsView.setObjectName("graphicsView")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(100, 30, 201, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(22)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setStyleSheet("color: rgb(255, 255, 255);")
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(90, 350, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton.setFont(font)
        self.pushButton.setStyleSheet(".QPushButton{\n"
"background-color : #EA4335;\n"
"border-radius : 12px;\n"
"color : balck ; \n"
"border : 2px solid #f44336;\n"
"}\n"
"\n"
".QPushButton:hover {\n"
"background-color : white;\n"
"color : black;\n"
"}")
        self.pushButton.setObjectName("pushButton")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(0, 0, 101, 81))
        self.label_2.setStyleSheet("image : url(logo.png)")
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(80, 120, 151, 151))
        self.label_3.setStyleSheet("image : url(qr-code-scan (1).png)")
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "VisionQuest"))
        MainWindow.setWindowIcon(QIcon("icon.jpg"))
        self.label.setText(_translate("MainWindow", "VisionQuest"))
        self.pushButton.setText(_translate("MainWindow", "Select Image"))
        self.pushButton.clicked.connect(self.upload_img)
    
    def upload_img(self):
        filename = QFileDialog.getOpenFileName()
        path = filename[0]
        path = str(path)
        print(path)
        img = cv2.imread(path)
        BGRimg = cv2.resize(img , (640,480))
        RGBimg = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        RGBimg = cv2.resize(RGBimg , (321,321))
        result = classifyimg(RGBimg)
        print(result)
        cv2.rectangle(BGRimg, (0,480) , (640,425) , (50 , 50 , 255) , -2)
        cv2.putText(BGRimg, f'Predicted : {result}',(20,460),cv2.FONT_HERSHEY_COMPLEX,
                    1,(255,255,255),1,cv2.LINE_AA)
        cv2.imshow("input ",BGRimg)
        cv2.waitKey(0)



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
