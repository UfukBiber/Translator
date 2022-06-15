from PyQt5.QtWidgets import QMainWindow, QWidget, QLabel, QLineEdit, QApplication, QPushButton, QGridLayout, QVBoxLayout
from PyQt5.QtCore import Qt
import sys
from Predict import Predict

class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.mainWidget = QWidget(self)
        self.hLayout = QGridLayout()
        self.vLayout = QVBoxLayout()
        self.setFixedSize(640, 480)
        self.InpText = QLineEdit("Enter Text")
        self.InpText.setAlignment(Qt.AlignCenter)
        self.OutText = QLabel()
        self.OutText.setText("None")
        self.OutText.setAlignment(Qt.AlignCenter)
        self.hLayout.addWidget(self.InpText, 0, 0, 1, 2)
        self.hLayout.addWidget(self.OutText, 0, 2, 1, 2)
        self.vLayout.addLayout(self.hLayout)
        self.button = QPushButton("Translate")
        self.button.clicked.connect(self.buttonClicked)
        self.vLayout.addWidget(self.button)
        self.mainWidget.setLayout(self.vLayout)
        self.setCentralWidget(self.mainWidget)
        self.show()
    
    def buttonClicked(self):
        InpText = self.InpText.text()
        OutText = Predict(InpText)
        self.OutText.setText(OutText)



if __name__ == "__main__":
    qapp = QApplication([])
    window = Window()
    qapp.exec_()
    sys.exit()