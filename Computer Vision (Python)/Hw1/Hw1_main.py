from PyQt5 import QtWidgets, uic, QtCore, QtGui
from ui import Hw1ui   #import the .py file which is turned from .ui file
import sys
import Q1_Q5_coding #Q1_1, Q1_2, Q1_3, Q1_4, Q2, Q3_1, Q3_2, Q4, Q5_1, Q5_2, Q5_3, Q5_4, Q5_5

#https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/584570/
#https://clay-atlas.com/blog/2019/08/27/pyqt5-%e5%9f%ba%e6%9c%ac%e6%95%99%e5%ad%b8-2-qlabel-qlineedit-qpushbutton/   怎麼設置按鈕...等

Ui_MainWindow = Hw1ui.Ui_MainWindow  #指定Ui_MainWindow 为main_menu文件下的Ui_MainWindow对象

class CoperQt(QtWidgets.QMainWindow,Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)  # 创建主界面对象
        Ui_MainWindow.__init__(self)#主界面对象初始化
        self.setupUi(self)  #配置主界面对象

        self.FindCorners.clicked.connect(self.Find_Corners)                       #Q1.1
        self.Instrinsic.clicked.connect(self.Find_IntrinsicMatrix)                #Q1.2
        self.Extrinsic.clicked.connect(self.Find_ExtrinsicMatrix)                 #Q1.3
        self.Distortion.clicked.connect(self.Find_DistortionMatrix)               #Q1.4
        self.AumentedReality.clicked.connect(self.Augmented_Reality)              #Q2
        self.Rotation.clicked.connect(self.RotationScalingTranslation)            #Q3.1
        self.Perspective.clicked.connect(self.Perspective_Transformation)         #Q3.2
        self.FindContour.clicked.connect(self.Find_Contour)                       #Q4

    def Find_Corners(self):                                                       #Q1.1
        Q1_Q5_coding.Findcorners()
    def Find_IntrinsicMatrix(self):                                               #Q1.2
        Q1_Q5_coding.Findintrinsicmatrix()
    def Find_ExtrinsicMatrix(self):                                               #Q1.3
        x = self.comboBox.currentText()
        Q1_Q5_coding.Findextrinsicmatrix(int(x))
    def Find_DistortionMatrix(self):                                              #Q1.4
        Q1_Q5_coding.Finddistortionmatrix()
    def Augmented_Reality(self):                                                  #Q2
        Q1_Q5_coding.Augmentedreality()
    def RotationScalingTranslation(self):                                         #Q3.1
        if not self.Angle.text() or not self.Scale.text() or not self.Tx.text() or not self.Ty.text():
            Q1_Q5_coding.ImageOriginaltransform()
        else :
            r = int(self.Angle.text())
            s = float(self.Scale.text())
            tx = int(self.Tx.text())
            ty = int(self.Ty.text())
            Q1_Q5_coding.Rotationscalingtranslation(r,s,tx,ty)
            self.Angle.clear()
            self.Scale.clear()
            self.Tx.clear()
            self.Ty.clear()
    def Perspective_Transformation(self):                                         #Q3.2
        Q1_Q5_coding.Perspectivetransformation()
    def Find_Contour(self):                                                       #Q4
        Q1_Q5_coding.Findcontour()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = CoperQt()#创建QT对象
    window.show()#QT对象显示
    app.exec_()



