from PyQt5 import QtWidgets, uic,QtCore,QtGui
import os
import sys
import Q1_Q3_coding

path = os.getcwd()
qtCreatorFile = path + os.sep+"Hw2.ui"
Ui_Hw2, QtBaseClass = uic.loadUiType(qtCreatorFile)
_translate = QtCore.QCoreApplication.translate

class MainUi(QtWidgets.QMainWindow, Ui_Hw2):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_Hw2.__init__(self)
        self.setupUi(self)

        self.Disparity.clicked.connect(self.Disparity_def)                     #Q1
        self.NCC.clicked.connect(self.NCC_def)                                 #Q2
        self.Keypoint.clicked.connect(self.Keypoint_def)                       #Q3.1
        self.MatchedKeypoints.clicked.connect(self.MatchedKeypoints_def)       #Q3.2

    def Disparity_def(self):                                                   #Q1
        Q1_Q3_coding.Disparity_Q1()

    def NCC_def(self):                                                         #Q2
        Q1_Q3_coding.NCC_Q2()

    def Keypoint_def(self):                                                    #Q3.1
        Q1_Q3_coding.Keypoint_Q3()

    def MatchedKeypoints_def(self):                                            #Q3.2
        Q1_Q3_coding.MatchedKeypoints_Q3()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainUi()
    window.show()
    app.exec_()


