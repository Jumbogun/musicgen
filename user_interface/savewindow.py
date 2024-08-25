# -*- coding: utf-8 -*-

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import wave
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.Qt import QMutex
class FirstWindow(QWidget):

    close_signal = pyqtSignal()
    def __init__(self, parent=None):
        # super这个用法是调用父类的构造函数
        # parent=None表示默认没有父Widget，如果指定父亲Widget，则调用之
        super(FirstWindow, self).__init__(parent)
        self.resize(100, 100)
        self.btn = QToolButton(self)
        self.btn.setText("click")
        self.sec = SaveWindow()
        self.btn.clicked.connect(self.sec.handle_click)
        #self.btn.clicked.connect(self.hide)
    def closeEvent(self, event):
        self.close_signal.emit()
        self.close()


class SaveWindow(QMainWindow):
    def __init__(self, parent=None):
        super(SaveWindow, self).__init__(parent)
        self.initUI()
        self.loadqss()
    def initUI(self):
        self.setWindowTitle('Save as') 
        self.resize(400, 100)
        desktop = QApplication.desktop()
        self.move(int(desktop.width()*0.4), int(desktop.height()*0.6))
        #self.setWindowIcon(QIcon('swag.ico'))
        self.setWindowOpacity(0.85) 
        
        #main layout
        self.main_layout   = QGridLayout()
        self.main_widget = QWidget()
        self.main_widget.setLayout(self.main_layout)
        
        #top layout
        self.top_widget = QWidget()
        self.top_widget.setObjectName('top_widget')
        self.layout_top = QGridLayout()
        self.top_widget.setLayout(self.layout_top)
        #bottom layout
        self.bottom_widget = QWidget()
        self.bottom_widget.setObjectName('bottom_widget')
        self.layout_bottom = QGridLayout()
        self.bottom_widget.setLayout(self.layout_bottom)
        
        
        self.name_input = QLineEdit() 
        self.name_input.setPlaceholderText("file name")
        self.layout_top.addWidget(self.name_input,  0, 0)
                
        self.btn_cancle = QPushButton("Cancle") # cancle btn
        self.btn_saveas_midi = QPushButton("Save as mid") 
        self.btn_saveas_wav = QPushButton("Save as wav") 
        
        self.btn_cancle.clicked.connect(self.handle_close)
        self.btn_saveas_midi.clicked.connect(self.saveas_midi)
        self.btn_saveas_wav.clicked.connect(self.saveas_wav)
        
        self.layout_bottom.addWidget(self.btn_cancle,       0, 0)
        self.layout_bottom.addWidget(self.btn_saveas_midi,  0, 1)
        self.layout_bottom.addWidget(self.btn_saveas_wav,   0, 2)
        
        
        self.main_layout.addWidget(self.top_widget,     0, 0, 1,  10) 
        self.main_layout.addWidget(self.bottom_widget,  1, 0, 1,  10)     
        self.setCentralWidget(self.main_widget)
        self.setAttribute(Qt.WA_TranslucentBackground)
    def pass_files(self, mid, wav):
        self.mid = mid
        self.wav = wav

    def handle_click(self):
        if not self.isVisible():
            self.show()   
    def saveas_midi(self):
        if (self.name_input.text() != ""):
            print(f"save as {self.name_input.text()}.mid" )
            self.mid.write(f"{self.name_input.text()}.mid" )
        else:
            print("null")
    def saveas_wav(self):
        if (self.name_input.text() != ""):
            print(f"save as {self.name_input.text()}.wav" )
            framerate = 44100
            print(self.wav.shape)
            wave_data = self.wav * 5000
            wave_data = wave_data.astype(np.short)
            f = wave.open(f"{self.name_input.text()}.wav", "wb")

            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(framerate)

            f.writeframes(wave_data.tostring())
            f.close()
        else:
            print("null")
    def handle_close(self):
        self.close()
    def mousePressEvent(self, event):
        if event.button()==Qt.LeftButton:
            self.m_flag=True
            self.m_Position=event.globalPos()-self.pos() #获取鼠标相对窗口的位置
            event.accept()
            self.setCursor(QCursor(Qt.OpenHandCursor))  #更改鼠标图标

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_flag:  
            self.move(QMouseEvent.globalPos()-self.m_Position)#更改窗口位置
            QMouseEvent.accept()

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag=False
        self.setCursor(QCursor(Qt.ArrowCursor))
         
    def loadqss(self):
        self.main_widget.setStyleSheet('''
            QWidget{
                border:none;
            }
        ''')
        self.bottom_widget.setStyleSheet('''
            QWidget#bottom_widget{
                background:white;
                border-bottom:1px solid darkGray;
                border-left:1px solid darkGray;
                border-right:1px solid darkGray;
                border-bottom-left-radius:10px;
                border-bottom-right-radius:10px;
                padding: none;
            }
        ''')
        self.btn_saveas_midi.setStyleSheet('''
            QPushButton{
                color:black;
                background:white;
                border-radius:5px;
                font-size:15px;
                font-weight:500;
                }
            QPushButton:hover{
                font-size:15px;
                font-weight:800;
            }
        ''')
        
        self.btn_saveas_wav.setStyleSheet('''
            QPushButton{
                color:black;
                background:white;
                border-radius:5px;
                font-size:15px;
                font-weight:500;
                }
            QPushButton:hover{
                font-size:15px;
                font-weight:800;
            }
        ''')
        self.btn_cancle.setStyleSheet('''
            QPushButton{
                color:black;
                background:white;
                border-radius:5px;
                font-size:15px;
                font-weight:500;
                }
            QPushButton:hover{
                color:red;
                font-size:15px;
                font-weight:800;
            }
        ''')
        self.top_widget.setStyleSheet('''
            QLineEdit{
                color:black;
                background:lightgrey;
                font-size:15px;
                font-weight:500;
            }
            QWidget#top_widget{
                background:lightgrey;
                border-top:1px solid darkGray;
                border-bottom:1px solid darkGray;
                border-left:1px solid darkGray;
                border-right:1px solid darkGray;
                border-top-left-radius:10px;
                border-top-right-radius:10px;
            }
        ''')
        self.layout_bottom.setSpacing(0)
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.main_layout.setSpacing(0)
if __name__ == "__main__":
    App = QApplication(sys.argv)
    ex = SaveWindow()
    ex.show()
    sys.exit(App.exec_())
