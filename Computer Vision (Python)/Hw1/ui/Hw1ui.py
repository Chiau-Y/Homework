# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\Hw1.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1068, 551)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.Cancel = QtWidgets.QPushButton(self.centralwidget)
        self.Cancel.setGeometry(QtCore.QRect(960, 460, 81, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.NoAntialias)
        self.Cancel.setFont(font)
        self.Cancel.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.Cancel.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.Cancel.setToolTipDuration(-1)
        self.Cancel.setAutoDefault(False)
        self.Cancel.setObjectName("Cancel")
        self.OK = QtWidgets.QPushButton(self.centralwidget)
        self.OK.setGeometry(QtCore.QRect(860, 460, 81, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.NoAntialias)
        self.OK.setFont(font)
        self.OK.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.OK.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.OK.setToolTipDuration(-1)
        self.OK.setAutoDefault(False)
        self.OK.setObjectName("OK")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(10, 30, 461, 241))
        self.groupBox_2.setObjectName("groupBox_2")
        self.groupBox = QtWidgets.QGroupBox(self.groupBox_2)
        self.groupBox.setGeometry(QtCore.QRect(210, 40, 221, 181))
        self.groupBox.setObjectName("groupBox")
        self.Extrinsic = QtWidgets.QPushButton(self.groupBox)
        self.Extrinsic.setGeometry(QtCore.QRect(30, 110, 166, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.NoAntialias)
        self.Extrinsic.setFont(font)
        self.Extrinsic.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.Extrinsic.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.Extrinsic.setToolTipDuration(-1)
        self.Extrinsic.setAutoDefault(False)
        self.Extrinsic.setObjectName("Extrinsic")
        self.comboBox = QtWidgets.QComboBox(self.groupBox)
        self.comboBox.setGeometry(QtCore.QRect(30, 70, 91, 31))
        self.comboBox.setInsertPolicy(QtWidgets.QComboBox.InsertAtBottom)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.Selectimage = QtWidgets.QTextEdit(self.groupBox)
        self.Selectimage.setEnabled(False)
        self.Selectimage.setGeometry(QtCore.QRect(20, 30, 121, 31))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        font.setKerning(True)
        self.Selectimage.setFont(font)
        self.Selectimage.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.Selectimage.setFrameShadow(QtWidgets.QFrame.Plain)
        self.Selectimage.setLineWidth(1)
        self.Selectimage.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.Selectimage.setCursorWidth(0)
        self.Selectimage.setObjectName("Selectimage")
        self.Distortion = QtWidgets.QPushButton(self.groupBox_2)
        self.Distortion.setGeometry(QtCore.QRect(20, 170, 166, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.NoAntialias)
        self.Distortion.setFont(font)
        self.Distortion.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.Distortion.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.Distortion.setToolTipDuration(-1)
        self.Distortion.setAutoDefault(False)
        self.Distortion.setObjectName("Distortion")
        self.FindCorners = QtWidgets.QPushButton(self.groupBox_2)
        self.FindCorners.setEnabled(True)
        self.FindCorners.setGeometry(QtCore.QRect(20, 30, 166, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.NoAntialias)
        self.FindCorners.setFont(font)
        self.FindCorners.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.FindCorners.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.FindCorners.setToolTipDuration(-1)
        self.FindCorners.setAutoDefault(False)
        self.FindCorners.setObjectName("FindCorners")
        self.Instrinsic = QtWidgets.QPushButton(self.groupBox_2)
        self.Instrinsic.setGeometry(QtCore.QRect(20, 100, 166, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.NoAntialias)
        self.Instrinsic.setFont(font)
        self.Instrinsic.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.Instrinsic.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.Instrinsic.setToolTipDuration(-1)
        self.Instrinsic.setAutoDefault(False)
        self.Instrinsic.setObjectName("Instrinsic")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(10, 290, 221, 161))
        self.groupBox_3.setObjectName("groupBox_3")
        self.AumentedReality = QtWidgets.QPushButton(self.groupBox_3)
        self.AumentedReality.setGeometry(QtCore.QRect(10, 50, 191, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.NoAntialias)
        self.AumentedReality.setFont(font)
        self.AumentedReality.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.AumentedReality.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.AumentedReality.setToolTipDuration(-1)
        self.AumentedReality.setAutoDefault(False)
        self.AumentedReality.setObjectName("AumentedReality")
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setGeometry(QtCore.QRect(250, 290, 221, 161))
        self.groupBox_4.setObjectName("groupBox_4")
        self.FindContour = QtWidgets.QPushButton(self.groupBox_4)
        self.FindContour.setGeometry(QtCore.QRect(10, 50, 191, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.NoAntialias)
        self.FindContour.setFont(font)
        self.FindContour.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.FindContour.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.FindContour.setToolTipDuration(-1)
        self.FindContour.setAutoDefault(False)
        self.FindContour.setObjectName("FindContour")
        self.groupBox_5 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_5.setGeometry(QtCore.QRect(480, 30, 271, 421))
        self.groupBox_5.setObjectName("groupBox_5")
        self.Perspective = QtWidgets.QPushButton(self.groupBox_5)
        self.Perspective.setGeometry(QtCore.QRect(30, 350, 211, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.NoAntialias)
        self.Perspective.setFont(font)
        self.Perspective.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.Perspective.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.Perspective.setToolTipDuration(-1)
        self.Perspective.setAutoDefault(False)
        self.Perspective.setObjectName("Perspective")
        self.groupBox_7 = QtWidgets.QGroupBox(self.groupBox_5)
        self.groupBox_7.setGeometry(QtCore.QRect(10, 30, 251, 301))
        self.groupBox_7.setFlat(False)
        self.groupBox_7.setCheckable(False)
        self.groupBox_7.setObjectName("groupBox_7")
        self.Rotation = QtWidgets.QPushButton(self.groupBox_7)
        self.Rotation.setGeometry(QtCore.QRect(20, 240, 211, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.NoAntialias)
        self.Rotation.setFont(font)
        self.Rotation.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.Rotation.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.Rotation.setToolTipDuration(-1)
        self.Rotation.setAutoDefault(False)
        self.Rotation.setObjectName("Rotation")
        self.groupBox_6 = QtWidgets.QGroupBox(self.groupBox_7)
        self.groupBox_6.setGeometry(QtCore.QRect(20, 30, 201, 201))
        self.groupBox_6.setObjectName("groupBox_6")
        self.textEdit_30 = QtWidgets.QTextEdit(self.groupBox_6)
        self.textEdit_30.setEnabled(False)
        self.textEdit_30.setGeometry(QtCore.QRect(20, 160, 51, 31))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        font.setKerning(True)
        self.textEdit_30.setFont(font)
        self.textEdit_30.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textEdit_30.setFrameShadow(QtWidgets.QFrame.Plain)
        self.textEdit_30.setLineWidth(1)
        self.textEdit_30.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.textEdit_30.setCursorWidth(0)
        self.textEdit_30.setObjectName("textEdit_30")
        self.textEdit_31 = QtWidgets.QTextEdit(self.groupBox_6)
        self.textEdit_31.setEnabled(False)
        self.textEdit_31.setGeometry(QtCore.QRect(10, 40, 61, 31))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        font.setKerning(True)
        self.textEdit_31.setFont(font)
        self.textEdit_31.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textEdit_31.setFrameShadow(QtWidgets.QFrame.Plain)
        self.textEdit_31.setLineWidth(1)
        self.textEdit_31.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.textEdit_31.setCursorWidth(0)
        self.textEdit_31.setObjectName("textEdit_31")
        self.textEdit_32 = QtWidgets.QTextEdit(self.groupBox_6)
        self.textEdit_32.setEnabled(False)
        self.textEdit_32.setGeometry(QtCore.QRect(10, 80, 61, 31))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        font.setKerning(True)
        self.textEdit_32.setFont(font)
        self.textEdit_32.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textEdit_32.setFrameShadow(QtWidgets.QFrame.Plain)
        self.textEdit_32.setLineWidth(1)
        self.textEdit_32.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.textEdit_32.setCursorWidth(0)
        self.textEdit_32.setObjectName("textEdit_32")
        self.textEdit_33 = QtWidgets.QTextEdit(self.groupBox_6)
        self.textEdit_33.setEnabled(False)
        self.textEdit_33.setGeometry(QtCore.QRect(20, 120, 51, 31))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        font.setKerning(True)
        self.textEdit_33.setFont(font)
        self.textEdit_33.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textEdit_33.setFrameShadow(QtWidgets.QFrame.Plain)
        self.textEdit_33.setLineWidth(1)
        self.textEdit_33.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.textEdit_33.setCursorWidth(0)
        self.textEdit_33.setObjectName("textEdit_33")
        self.textEdit_34 = QtWidgets.QTextEdit(self.groupBox_6)
        self.textEdit_34.setEnabled(False)
        self.textEdit_34.setGeometry(QtCore.QRect(140, 160, 51, 31))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        font.setKerning(True)
        self.textEdit_34.setFont(font)
        self.textEdit_34.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textEdit_34.setFrameShadow(QtWidgets.QFrame.Plain)
        self.textEdit_34.setLineWidth(1)
        self.textEdit_34.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.textEdit_34.setCursorWidth(0)
        self.textEdit_34.setObjectName("textEdit_34")
        self.textEdit_35 = QtWidgets.QTextEdit(self.groupBox_6)
        self.textEdit_35.setEnabled(False)
        self.textEdit_35.setGeometry(QtCore.QRect(140, 40, 51, 41))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        font.setKerning(True)
        self.textEdit_35.setFont(font)
        self.textEdit_35.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textEdit_35.setFrameShadow(QtWidgets.QFrame.Plain)
        self.textEdit_35.setLineWidth(1)
        self.textEdit_35.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.textEdit_35.setCursorWidth(0)
        self.textEdit_35.setObjectName("textEdit_35")
        self.textEdit_36 = QtWidgets.QTextEdit(self.groupBox_6)
        self.textEdit_36.setEnabled(False)
        self.textEdit_36.setGeometry(QtCore.QRect(140, 120, 51, 31))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        font.setKerning(True)
        self.textEdit_36.setFont(font)
        self.textEdit_36.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textEdit_36.setFrameShadow(QtWidgets.QFrame.Plain)
        self.textEdit_36.setLineWidth(1)
        self.textEdit_36.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.textEdit_36.setCursorWidth(0)
        self.textEdit_36.setObjectName("textEdit_36")
        self.Angle = QtWidgets.QLineEdit(self.groupBox_6)
        self.Angle.setGeometry(QtCore.QRect(70, 40, 61, 21))
        self.Angle.setObjectName("Angle")
        self.Scale = QtWidgets.QLineEdit(self.groupBox_6)
        self.Scale.setGeometry(QtCore.QRect(70, 80, 61, 21))
        self.Scale.setObjectName("Scale")
        self.Tx = QtWidgets.QLineEdit(self.groupBox_6)
        self.Tx.setGeometry(QtCore.QRect(70, 120, 61, 21))
        self.Tx.setObjectName("Tx")
        self.Ty = QtWidgets.QLineEdit(self.groupBox_6)
        self.Ty.setGeometry(QtCore.QRect(70, 160, 61, 21))
        self.Ty.setObjectName("Ty")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(450, 620, 47, 12))
        self.label.setObjectName("label")
        self.groupBox_8 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_8.setGeometry(QtCore.QRect(760, 30, 301, 421))
        self.groupBox_8.setObjectName("groupBox_8")
        self.TrainImage = QtWidgets.QPushButton(self.groupBox_8)
        self.TrainImage.setGeometry(QtCore.QRect(20, 60, 261, 41))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.TrainImage.setFont(font)
        self.TrainImage.setObjectName("TrainImage")
        self.Hyperparameters = QtWidgets.QPushButton(self.groupBox_8)
        self.Hyperparameters.setGeometry(QtCore.QRect(20, 120, 261, 41))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.Hyperparameters.setFont(font)
        self.Hyperparameters.setObjectName("Hyperparameters")
        self.TrainEpoch = QtWidgets.QPushButton(self.groupBox_8)
        self.TrainEpoch.setGeometry(QtCore.QRect(20, 180, 261, 41))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.TrainEpoch.setFont(font)
        self.TrainEpoch.setObjectName("TrainEpoch")
        self.TrainingResult = QtWidgets.QPushButton(self.groupBox_8)
        self.TrainingResult.setGeometry(QtCore.QRect(20, 240, 261, 41))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.TrainingResult.setFont(font)
        self.TrainingResult.setObjectName("TrainingResult")
        self.Inference = QtWidgets.QPushButton(self.groupBox_8)
        self.Inference.setGeometry(QtCore.QRect(20, 340, 261, 41))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.Inference.setFont(font)
        self.Inference.setObjectName("Inference")
        self.ImageIndex = QtWidgets.QLineEdit(self.groupBox_8)
        self.ImageIndex.setGeometry(QtCore.QRect(140, 300, 81, 21))
        self.ImageIndex.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.ImageIndex.setAutoFillBackground(False)
        self.ImageIndex.setInputMethodHints(QtCore.Qt.ImhNone)
        self.ImageIndex.setText("")
        self.ImageIndex.setFrame(True)
        self.ImageIndex.setEchoMode(QtWidgets.QLineEdit.Normal)
        self.ImageIndex.setCursorPosition(0)
        self.ImageIndex.setAlignment(QtCore.Qt.AlignCenter)
        self.ImageIndex.setDragEnabled(False)
        self.ImageIndex.setCursorMoveStyle(QtCore.Qt.LogicalMoveStyle)
        self.ImageIndex.setClearButtonEnabled(True)
        self.ImageIndex.setObjectName("ImageIndex")
        self.textEdit_44 = QtWidgets.QTextEdit(self.groupBox_8)
        self.textEdit_44.setEnabled(False)
        self.textEdit_44.setGeometry(QtCore.QRect(20, 300, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        font.setKerning(True)
        self.textEdit_44.setFont(font)
        self.textEdit_44.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textEdit_44.setFrameShadow(QtWidgets.QFrame.Plain)
        self.textEdit_44.setLineWidth(1)
        self.textEdit_44.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.textEdit_44.setCursorWidth(0)
        self.textEdit_44.setObjectName("textEdit_44")
        self.label_2 = QtWidgets.QLabel(self.groupBox_8)
        self.label_2.setGeometry(QtCore.QRect(230, 300, 61, 21))
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1068, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Cancel.setText(_translate("MainWindow", "Cancel"))
        self.OK.setText(_translate("MainWindow", "OK"))
        self.groupBox_2.setTitle(_translate("MainWindow", "1. Calibration"))
        self.groupBox.setTitle(_translate("MainWindow", "1.3 Extrinsic"))
        self.Extrinsic.setText(_translate("MainWindow", "1.3 Extrinsic"))
        self.comboBox.setItemText(0, _translate("MainWindow", "1"))
        self.comboBox.setItemText(1, _translate("MainWindow", "2"))
        self.comboBox.setItemText(2, _translate("MainWindow", "3"))
        self.comboBox.setItemText(3, _translate("MainWindow", "4"))
        self.comboBox.setItemText(4, _translate("MainWindow", "5"))
        self.comboBox.setItemText(5, _translate("MainWindow", "6"))
        self.comboBox.setItemText(6, _translate("MainWindow", "7"))
        self.comboBox.setItemText(7, _translate("MainWindow", "8"))
        self.comboBox.setItemText(8, _translate("MainWindow", "9"))
        self.comboBox.setItemText(9, _translate("MainWindow", "10"))
        self.comboBox.setItemText(10, _translate("MainWindow", "11"))
        self.comboBox.setItemText(11, _translate("MainWindow", "12"))
        self.comboBox.setItemText(12, _translate("MainWindow", "13"))
        self.comboBox.setItemText(13, _translate("MainWindow", "14"))
        self.comboBox.setItemText(14, _translate("MainWindow", "15"))
        self.Selectimage.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-weight:600;\">Select image</span></p></body></html>"))
        self.Distortion.setText(_translate("MainWindow", "1.4 Distortion"))
        self.FindCorners.setText(_translate("MainWindow", "1.1 Find Corners"))
        self.Instrinsic.setText(_translate("MainWindow", "1.2 Instrinsic"))
        self.groupBox_3.setTitle(_translate("MainWindow", "2 Augmented Reality"))
        self.AumentedReality.setText(_translate("MainWindow", "2.1 Augmented Reality"))
        self.groupBox_4.setTitle(_translate("MainWindow", "4. Find Contour"))
        self.FindContour.setText(_translate("MainWindow", "4.1 Find Contour"))
        self.groupBox_5.setTitle(_translate("MainWindow", "3. Image Transformation"))
        self.Perspective.setText(_translate("MainWindow", "3.2 Perspective Transform"))
        self.groupBox_7.setTitle(_translate("MainWindow", "3.1 Rot, scale, Translate"))
        self.Rotation.setText(_translate("MainWindow", "3.1 Rotation, scaling, translation"))
        self.groupBox_6.setTitle(_translate("MainWindow", "Parameters"))
        self.textEdit_30.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:8pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-weight:600;\">Ty:</span></p></body></html>"))
        self.textEdit_31.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:8pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-weight:600;\">Angle:</span></p></body></html>"))
        self.textEdit_32.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:8pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-weight:600;\">Scale:</span></p></body></html>"))
        self.textEdit_33.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:8pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-weight:600;\">Tx:</span></p></body></html>"))
        self.textEdit_34.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:8pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-weight:600;\">pixel</span></p></body></html>"))
        self.textEdit_35.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:8pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-weight:600;\">deg</span></p></body></html>"))
        self.textEdit_36.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:8pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-weight:600;\">pixel</span></p></body></html>"))
        self.label.setText(_translate("MainWindow", "TextLabel"))
        self.groupBox_8.setTitle(_translate("MainWindow", "Train Cifar-10 Classifier Using LeNet-5 "))
        self.TrainImage.setText(_translate("MainWindow", "5.1 Show Train Image"))
        self.Hyperparameters.setText(_translate("MainWindow", "5.2 Show Hyperparameters"))
        self.TrainEpoch.setText(_translate("MainWindow", "5.3 Train 1 Epoch"))
        self.TrainingResult.setText(_translate("MainWindow", "5.4 Show Training Result"))
        self.Inference.setText(_translate("MainWindow", "5.5 Inference"))
        self.textEdit_44.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:8pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:9pt; font-weight:600;\">Test Image Index:</span></p></body></html>"))
        self.label_2.setText(_translate("MainWindow", "(0~9999)"))

