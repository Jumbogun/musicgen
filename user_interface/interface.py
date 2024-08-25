import numpy as np
import matplotlib.pyplot as plt
import pypianoroll as pr
import pretty_midi
import os
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.models as models  
import torchvision.transforms as transforms
from torch.nn.utils import spectral_norm
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.nn import Parameter
import shutil
import time
import qtawesome
import sys
import pyaudio
import wave
from pyaudio import PyAudio
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.Qt import QMutex
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FC
from util import *
from interface import *
from model import *
from savewindow import *
from threadgen import *
from threadwav import *
from threadaudio import *
from threadcanvas import *

class MCheckBox(QCheckBox):
    def __init__(self, idx, param):
        QCheckBox.__init__(self, param)
        #设置标题与初始大小
        self.id = idx

class Box(QMainWindow):
    def __init__(self, args,  gen_model, pixelcnn, device):
        self.gen_model = gen_model.eval()
        self.pixelcnn = pixelcnn.eval()
        self.device = device
        
        self.args = args
        #unload arg
        self.program_nums     = args["program_nums"]                  # Editable
        self.is_drum          = args["is_drum"]                       # Editable
        self.tempo            = args["tempo"]                         # Editable
        self.n_tracks         = args["n_tracks"]
        self.n_pitches        = args["n_pitches"]
        self.n_bars           = args["n_bars"]                      # Editable
        self.lowest_pitch     = args["lowest_pitch"]
        self.n_measures       = args["n_measures"]
        self.beat_resolution  = args["beat_resolution"]               # 
        self.batch_size       = args["batch_size"]
        self.decay            = args["decay"]
        self.commitment_cost  = args["commitment_cost"]
        self.latent_w         = args["latent_w"]
        self.latent_h         = args["latent_h"]
        self.latent_dim       = args["latent_dim"]
        self.embedding_dim    = args["embedding_dim"]
        self.num_embeddings   = args["num_embeddings"]
        
        self.resolution = 4 * args["n_measures"] * args["beat_resolution"]  #64 # bar resolution
        self.tracks_selection = [True, True, True, True, True]        # Editable          
        
        self.n_dis_bars = 2                                               # how many bar the canvas show
        self.dh, self.dw = self.n_pitches * self.n_tracks,  384 * self.n_dis_bars 
        
        self.pr = None                                                #  piano roll    (stretch to 24pixel per beat )
        self.im = None                                                #  piano roll visualize image (merge and color all track and  padded the time step axis with zeros)
        self.im_bg = None                                             #  canvas background
        self.im_w = -1                                                #  image width (= self.pr.shape[1] =self.im.shape[0])
        self.wav = None
        self.waves = None
        self.pm									 = None
        self.scale_factor = 24 // self.beat_resolution                # 
        self.time_steps = -1                                          #  time_steps should always be im_w // scale_factor
        
        self.audio_thread  = AudioThread(self)
        self.canvas_thread = CanvasThread(self) 
        self.gen_thread    = GenThread(args, gen_model, pixelcnn, device)
        self.gen_thread.donesignal.connect(self.post_gen_pr_event)
        
        self.wav_thread    = WavThread()
        self.wav_thread.donesignal.connect(self.post_gen_wav_event)
        
        self.is_playing = False                                       # flag for play control
        self.is_sliderPressed = False                                 # 
        self.cur_pixel = -1                                           # cursor on piano roll(used by both audio and canvas )

        super().__init__()
        self.initUI()
        self.loadqss()
    def initUI(self):
        
        self.setWindowTitle('DEMO')
        self.resize(1800,1000)
        self.setWindowIcon(QIcon('./image/swag.ico'))
        self.thr = 0.5
        self.setWindowOpacity(0.97) 
        self.setAttribute(Qt.WA_TranslucentBackground) 
        self.savewin = SaveWindow()

        #main layout
        self.main_layout   = QGridLayout()
        self.main_widget = QWidget()
        self.main_widget.setLayout(self.main_layout)
        
        #menu layout
        self.menu_widget = QWidget()
        self.menu_widget.setObjectName('menu_widget')
        self.layout_menu = QGridLayout()
        self.menu_widget.setLayout(self.layout_menu)
        #right layout
        self.right_widget = QWidget()
        self.right_widget.setObjectName('right_widget')
        self.layout_right = QGridLayout()
        self.right_widget.setLayout(self.layout_right)
        
        
        #track config layout
        self.tracks = QWidget()
        self.tracks.setObjectName('tracks')
        self.layout_tracks = QGridLayout()
        self.tracks.setLayout(self.layout_tracks)
	



	#adding widget for each track        
        self.track_checkboxL = []
        self.track_comboboxL = []
        for i in range(self.n_tracks):
            track = QWidget()
            not_drum = not self.is_drum[i]
            layout_track_cur = QGridLayout()
            track.setLayout(layout_track_cur)
            track_color = np.array([[254, 90, 187], [66, 254, 106], [235, 254, 90], [67, 199, 254], [167, 25, 25]], np.int32) 
            #checkbox
            checkbox = MCheckBox(i, self)
            checkbox.setChecked(True)
            checkbox.setStyleSheet("QWidget:checked{background:rgb" + f"({track_color[i][0]:d}, {track_color[i][1]:d}, {track_color[i][2]:d})" + "}")
            layout_track_cur.addWidget(checkbox, 0, 2, 1, 1)
            checkbox.stateChanged.connect(self.checkbox_conn)
            self.track_checkboxL.append(checkbox)

            #combobox
            combobox = QComboBox(self)
            layout_track_cur.addWidget(combobox, 0, 0, 1, 2)
            combobox.addItems(self.args["track_code"][not_drum].values())
            combobox.setFixedWidth(100)
            #combobox.setStyleSheet("QWidget{background:rgb" + f"({track_color[i][0]:d}, {track_color[i][1]:d}, {track_color[i][2]:d})" + "}")
            combobox.setStyleSheet("QWidget{selection-background-color:rgba" + f"({track_color[i][0]:d}, {track_color[i][1]:d}, {track_color[i][2]:d}, 143)" + "}")
            combobox.setCurrentText(self.args["track_code"][not_drum][self.program_nums[i]])
            combobox.currentIndexChanged[str].connect(self.reset_tracks_progress) # 条目发生改变，发射信号，传递条目内容
            #combobox.currentIndexChanged[int].connect(self.print_value)  # 条目发生改变，发射信号，传递条目索引
            #combobox.highlighted[str].connect(self.print_value)  # 在下拉列表中，鼠标移动到某个条目时发出信号，传递条目内容
            #combobox.highlighted[int].connect(self.print_value) 
            self.track_comboboxL.append(combobox)

            self.layout_tracks.addWidget(track, i, 0, 1, 1)
        
                    
        
        #init canvas
        self.init_canvas()
        
        self.init_progressbar()
        
        self.content = QLabel(self) 
        
        self.btn_close = QPushButton("x") # close btn
        self.btn_mini  = QPushButton("-")  # minimize btn
        
        self.btn_close.clicked.connect(self.close_event)
        self.btn_mini.clicked.connect(self.mini_event)
        self.layout_menu.addWidget(self.btn_mini,  0, 0,  1, 1)
        self.layout_menu.addWidget(self.btn_close, 0, 1,  1, 1)
        self.layout_menu.addWidget(self.tracks,    1, 0, 8, 4)
        
        self.bars_input = QLineEdit()
        self.bars_input.setValidator(QIntValidator(self.n_dis_bars, 36))
        self.bars_input.setPlaceholderText("number of bars")
        self.bars_input.setText(str(self.n_bars))
        self.tempo_input = QLineEdit()
        self.tempo_input.setValidator(QIntValidator(20, 240))
        self.tempo_input.setPlaceholderText("tempo")
        self.tempo_input.setText(str(self.tempo))
        
        self.layout_menu.addWidget(self.bars_input,    9, 2, 1, 2)
        #self.layout_menu.addWidget(self.tempo_input,  10, 2, 1, 2)
        
        self.layout_right.addWidget(self.canvas ,            0, 0,   8, 10)
        self.layout_right.addWidget(self.process_bar,        8, 0,   2, 10)
        self.layout_right.addWidget(self.playconsole_widget, 10, 0,  2, 10)
        
        self.main_layout.addWidget(self.menu_widget,   0, 0, 15,  3) 
        self.main_layout.addWidget(self.right_widget,  0, 3, 14,  8)     
        self.setCentralWidget(self.main_widget)
        
        desktop = QApplication.desktop()
        print(int(desktop.width()*0.1), int(desktop.height()*0.7))
        self.move(int(desktop.width()*0.1), int(desktop.height()*0.1))
        self.show()
    def init_progressbar(self):
        #setup progress bar
        self.process_bar = QSlider(Qt.Horizontal) 
        self.process_bar.setObjectName('process_bar') 
        #self.process_bar.setFixedHeight(4) 
        self.process_bar.setValue(0)
      
        self.process_bar.setMinimum(0)
        self.process_bar.setMaximum(0)
        self.process_bar.setSingleStep(1)
    
        self.process_bar.sliderReleased.connect(self.process_bar_release)
        self.process_bar.sliderPressed.connect(self.process_bar_pressed)
        
        #setup control widget
        self.playconsole_widget = QWidget()  
        self.playconsole_layout = QGridLayout()  
        self.playconsole_widget.setObjectName('playconsole_widget') 
        self.playconsole_widget.setLayout(self.playconsole_layout)

        self.btn_generate = QPushButton(qtawesome.icon('fa.headphones', color='#191919', font=16), "")
        self.btn_generate.setIconSize(QSize(40, 40))
        self.btn_generate.clicked.connect(self.generate_control)
        self.playconsole_layout.addWidget(self.btn_generate, 0, 0)

        self.btn_play = QPushButton(qtawesome.icon('fa.play-circle', color='#191919', font=18), "")
        self.btn_play.setIconSize(QSize(50, 50))
        self.playconsole_layout.addWidget(self.btn_play, 0, 2)
        self.btn_play.clicked.connect(self.play_control)

        self.btn_save = QPushButton(qtawesome.icon('fa.download', color='#191919', font=16), "")
        self.btn_save.setIconSize(QSize(40, 40))
        self.btn_save.clicked.connect(self.save_control)
        self.playconsole_layout.addWidget(self.btn_save, 0, 4)        

        self.playconsole_layout.setAlignment(Qt.AlignCenter) 
    def save_control(self):
        #pop up save window
        if((self.pm != None) or (self.waves != None)):
            self.savewin.pass_files(self.pm, self.wav)
            self.savewin.handle_click()
    def generate_control(self):
        print("loading")
        self.play_pause()
        self.btn_generate.setEnabled(False)                     # prevent duplicate  operator 
        self.btn_play.setEnabled(False)                         #  
        self.btn_play.setIcon(qtawesome.icon('fa.spinner', color='#F21862', font=18))
        
        n_bars = self.bars_input.text()
        if(n_bars != ""):
            self.set_n_bars(int(n_bars))
        self.gen_thread.generate(self.n_bars, self.tracks_selection)

    def post_gen_pr_event(self):
        self.pr = self.gen_thread.get_output()
        self.im_w = self.pr.shape[0] * self.pr.shape[1]
        self.time_steps =self.im_w // self.scale_factor
        self.wav_thread.generate(self.pr, self.time_steps, self.program_nums, self.is_drum, self.tempo)
        
        self.im = self.pr2im(self.pr)
        self.im = np.concatenate([self.im, np.zeros([self.im.shape[0], self.dw, 3], np.int32)], axis = 1)
        
        print(self.pr.shape, self.time_steps, self.im.shape)
        self.process_bar.setMaximum(self.time_steps)
        self.set_cur_pixel(0, False)
    def post_gen_wav_event(self):        
        
        self.waves, self.pm, self.wav = self.wav_thread.get_output()
        self.audio_thread.set_waves(self.waves)
        self.audio_thread.set_cur_pixel(self.cur_pixel)
        self.btn_generate.setEnabled(True)                     # prevent duplicate  operator 
        self.btn_play.setEnabled(True)                         #  
        self.play_pause()
    
    def pr2im(self, pr):

        #track_color = np.array([[58, 37, 167], [23, 102, 36], [231, 231, 49], [164, 25, 25], [0, 214, 255]], np.int32) 
        track_color = np.array([[254, 90, 187], [66, 254, 106], [235, 254, 90], [67, 199, 254], [167, 25, 25]], np.int32) 
         
        im = (pr > 0.5).astype(np.int32)
        im = im.reshape(im.shape[0] * im.shape[1], im.shape[2], im.shape[3])
        ret = np.zeros([im.shape[0], im.shape[1], 3], np.int32)
        for i in range(self.n_tracks):
            tmp = np.stack([im[..., i]] * 3, axis = -1) * track_color[i]
            ret += tmp
        ret = np.minimum(ret, 255).transpose(1, 0, 2)
        

        self.im_bg = np.ones([ret.shape[0], self.dw, 3], np.int32) *  np.array([219, 219, 229])
        
        for i in range(1, self.n_dis_bars * self.n_measures):
            self.im_bg[:, 96 * i] = 59
        return ret
        
    def play_start(self):
        if((self.pr is None) or (self.waves is None)):
            print("please generate new sond first") # TODO
            return
        self.is_playing = True
        self.btn_play.setIcon(qtawesome.icon('fa.pause-circle', color='#F21862', font=18))
        print("start")
        #self.timer.start()
        if(self.audio_thread is not None):
            self.audio_thread.set_cur_pixel(self.cur_pixel)
            self.audio_thread.set_waves(self.waves)
        else: 
            # Idealy this won't happen
            pass
            
        self.audio_thread.start()
        
    def play_pause(self):
        if(self.audio_thread is not None):
            self.audio_thread.stop()
        self.is_playing = False
        print("pause")
        self.btn_play.setIcon(qtawesome.icon('fa.play-circle', color='#191919', font=16))
    def play_control(self):
        if(self.is_playing):
            self.play_pause()
        else:
            self.play_start()     
            
    def set_cur_pixel(self, v, is_sliderPressed):
        print("set_cur_pixel", self.cur_pixel, self.time_steps)
        if(self.cur_pixel >= self.time_steps - 1):
            self.play_pause()
            self.cur_pixel = 0
            v = 0
        
        self.cur_pixel = v
        self.update_canvas()      
        if(not is_sliderPressed):
            if(self.audio_thread is not None):
                self.audio_thread.stop()
                #time.sleep(0.2)
                print("set a cur", v)
                self.audio_thread.set_cur_pixel(v)
                print(self.is_playing)
                if(self.is_playing):
                    print("////////")
                    self.audio_thread.start()        
            self.process_bar.setValue(v)
    def update_cur_pixel(self, v, is_sliderPressed):
        #print("update_cur_pixel", self.cur_pixel, self.time_steps)
        if(self.cur_pixel >= self.time_steps - 1):
            #print(self.cur_pixel, ">= self.time_step in in in in in " )
            self.play_pause()
            self.cur_pixel = 0
            v = 0
        
        self.cur_pixel = v
        self.update_canvas()   
        if(not is_sliderPressed):    
            self.process_bar.setValue(v)
    def inc_cur_pixel(self):
        #s = time.time()
        self.update_cur_pixel(self.cur_pixel + 1, self.is_sliderPressed)
        #e = time.time()
        #print("e - s", e - s)
    def process_bar_pressed(self):
        self.is_sliderPressed = True
    def process_bar_release(self):
        v = self.process_bar.value()
        #print('current slider value=%s'%v)
        self.set_cur_pixel(v, False)
        self.is_sliderPressed = False
        pass
    def set_n_bars(self, n_bars):
        if(n_bars< self.n_dis_bars):
            n_bars = self.n_dis_bars
        self.n_bars = n_bars

    def init_canvas(self):
        plt.axis('tight')
        self.fig, self.ax = plt.subplots(figsize=(20, 60), tight_layout = True)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_visible(False)
        self.fig.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
        self.ax.axes.set_axis_off()
        self.canvas = FC(self.fig)
        
        self.canvas.setObjectName('canvas')
        # init is the displaying piano roll
        self.ax.cla()

        
        self.ax.imshow(init_logo(), aspect='auto')
        #self.ax.imshow(self.fake[:, :, :, 0].reshape(self.fake.shape[0] * self.fake.shape[1], self.fake.shape[2],).transpose([1, 0]) > self.thr)
        self.canvas.draw() 
        

      
    def get_spp(self):
        #calculate how many second per pixel from tempo
        spp = 60000 / self.tempo / 24
        return spp
    def init_lt(self):
        self.ntimer = QTimer()
        self.ntimer.setInterval(1000)
        self.ntimer.start()
        self.ntimer.timeout.connect(self.update_time)
    def update_time(self):
        #self.content.setText(time.strftime("%X", time.localtime()))
        pass
    def init_timer(self):
        self.timer = QTimer()
        #print("spp", self.get_spp())
        self.timer.setInterval(self.get_spp())
        self.timer.timeout.connect(self.inc_cur_pixel)
        

    def update_canvas(self):
        #print("pic  :  ", self.cur_pixel)
        start = self.cur_pixel * self.scale_factor
        end = start + self.dw
        #self.display_im = #np.maximum(self.im[:,  start: end] , self.im_bg)
        self.display_im = self.im_bg * (self.im[:,  start: end] == 0)
        
        self.display_im = np.maximum(self.im[:,  start: end] , self.display_im)
        
        self.ax.cla()
        self.ax.imshow(self.display_im, aspect='auto',  interpolation='nearest')#,  interpolation='nearest'
        self.canvas.draw() 
        pass
        
    #TODO
    def reset_tracks_progress(self, i):
        self.play_pause()
        program = []
        for i in range(self.n_tracks):
            index = int(self.track_comboboxL[i].currentText().split()[0]) 
            program.append(index)
        #print(program)
        self.program_nums = program
        
        self.btn_generate.setEnabled(False)                     # prevent duplicate  operator 
        self.btn_play.setEnabled(False)                         #  
        self.btn_play.setIcon(qtawesome.icon('fa.spinner', color='#F21862', font=18))
        self.wav_thread.generate(self.pr, self.time_steps, self.program_nums, self.is_drum, self.tempo)
        
    def checkbox_conn(self):
        text = ""
        for i in range(self.n_tracks):
            self.tracks_selection[i] = self.track_checkboxL[i].isChecked()
            choice = str(self.track_checkboxL[i].id) if self.track_checkboxL[i].isChecked() else ''
            #print(i, self.track_checkboxL[i].isChecked())
            text += choice
        #print(text)
        self.content.setText(text)
    @pyqtSlot()
    def close_event(self):
        self.close()
    @pyqtSlot()
    def mini_event(self):
        self.showMinimized()
        
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
        #self.btn_close.setFixedSize(15,15)
        #self.btn_mini.setFixedSize(15,15)
        self.btn_close.setStyleSheet('''
            QPushButton{
                height:20px;
                background:#F76677;
                border-bottom-left-radius:2px;
                border-bottom-right-radius:2px;
                border-top-right-radius:5px;
                color:white;
            }
            QPushButton:hover{
                background:red;
            }''')
        self.btn_mini.setStyleSheet('''
            QPushButton{
                height:20px;
                background:#6DDF6D;
                border-bottom-left-radius:2px;
                border-bottom-right-radius:2px;
                border-top-left-radius:5px;
                color:white;
            }
            QPushButton:hover{
                background:green;
            }''')
        #selection-background-color: rgba(193, 242, 232, 143);
        self.menu_widget.setStyleSheet('''


            QCheckBox{
                width : 15px;
                height : 10px;
                padding: 0px;	
                border-radius :10px;
                border : 5px;
                background: grey;
            }
            QCheckBox:checked{
                width : 15px;
                height : 10px;
                padding: 0px;
                border-radius :10px;
                border : 5px;
            }

            QCheckBox::indicator:unchecked{
                background:white;
                width : 20px;
                height : 20px;
                padding: 0px;
                border-radius :10px;
                border : 2px solid white;
            }
            QCheckBox::indicator:checked{
                background:white;
                width : 20px;
                height : 20px;
                padding: 0px;
                border-radius :10px;
                border : 2px solid white;
                position: relative;
                left: 21px;
            }
            QComboBox{
                color:black;
                background:white;
                font-size:13px;
                font-weight:700;
	        selection-color: rgb(95, 166, 186);
                border:1px solid gray;
                border-radius:5px;
                padding:2px 4px;
            }
            QComboBox::drop-down{
                background:rgba(0, 0, 0, 0);;
                border:0px;
                height:10px;
                border-radius:5px;
            }

            QWidget#menu_widget{
                border-image: url(image/abglg2.png);
                border-top:1px solid white;
                border-bottom:1px solid white;
                border-left:1px solid white;
                border-top-left-radius:10px;
                border-bottom-left-radius:10px;
                border-bottom-right-radius:10px;
            }
            ''')
        self.right_widget.setStyleSheet('''
            QWidget#right_widget{
                background:white;
                border-top:1px solid darkGray;
                border-bottom:1px solid darkGray;
                border-right:1px solid darkGray;
                border-top-right-radius:10px;
                border-bottom-right-radius:10px;
            }
        ''')        
        self.main_widget.setStyleSheet('''
            QWidget#right_widget{
                border:none;
            }
        ''')
        self.process_bar.setStyleSheet('''
             QSlider::add-page:Horizontal
             {     
                background-color: rgb(87, 97, 106);
                height:6px;
             }
             QSlider::sub-page:Horizontal 
            {
                background-color:qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(231,80,229, 255), stop:1 rgba(7,208,255, 255));
                height:4px;
             }
            QSlider::groove:Horizontal 
            {
                background:transparent;
                height:6px;
            }
            QSlider::handle:Horizontal 
            {
                height: 40px;
                width: 15px;
                border-image: url(image/4.png);
                margin: -5 0px; 
            }
        ''')

        self.playconsole_widget.setStyleSheet('''
            QPushButton{
                border:none;
            }
        ''')
        self.bars_input.setStyleSheet('''
            QLineEdit{
                color:black;
                background:white;
                font-size:15px;
                font-weight:500;
                
                border-top:1px solid darkGray;
                border-bottom:1px solid darkGray;
                border-left:1px solid darkGray;
                border-right:1px solid darkGray;
                border-radius:5px;
            }
        ''')
        self.tempo_input.setStyleSheet('''
            QLineEdit{
                color:black;
                background:white;
                font-size:15px;
                font-weight:500;
                
                border-top:1px solid darkGray;
                border-bottom:1px solid darkGray;
                border-left:1px solid darkGray;
                border-right:1px solid darkGray;
                border-radius:5px;
            }
        ''')
      
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.layout_menu.setSpacing(0)
        self.main_layout.setSpacing(0)



def init_logo():
    lena = mpimg.imread('image/swagbg.png') 
    lena.shape

    return lena
        

