"""Tkinter desktop interface sharing the existing recognition and speech modules."""
import time
import tkinter as tk
from tkinter import ttk
import cv2
from config import settings
from src.hand_tracking.hand_detector import HandDetector, draw_hand_results
from src.hand_tracking.landmark_processor import create_feature_vector
from src.recognition.predictor import LivePredictor
from src.recognition.message_buffer import MessageBuffer
from src.speech.local_speech import LocalSpeech

BG='#F3F1EB'; PANEL='#FFFFFF'; TEXT='#202923'; MUTED='#788078'; ACCENT='#D9F27E'
BORDER='#E2E5DD'; DARK='#172C25'; CORAL='#EC987E'

class SinabtanayApp:
    def __init__(self, root, model_dir, camera_index=0, threshold=.70, speech_enabled=True):
        self.root=root
        self.camera=self.detector=None
        self.after_id=None
        self.closed=False
        self.last_timestamp=-1
        self.current_label=None
        self.speech=LocalSpeech(speech_enabled)
        self.predictor=LivePredictor(model_dir,threshold,7,5)
        self.message=MessageBuffer(self.predictor.classes)
        self.root.title('Sinabtanay')
        self.root.geometry('1240x820'); self.root.minsize(1040,800)
        self.root.configure(bg=BG)
        self.root.protocol('WM_DELETE_WINDOW',self.close)
        self.build_ui()
        self.root.bind('<KeyPress>',self.keypress)
        try:
            self.detector=HandDetector()
            self.camera=cv2.VideoCapture(camera_index)
            if not self.camera.isOpened():
                raise RuntimeError('Cannot open webcam. Close other camera apps or try --camera-index 1.')
            self.after_id=self.root.after(0,self.tick)
        except Exception:
            self.close()
            raise

    def label(self,parent,text,**kwargs):
        options=dict(bg=parent.cget('bg'),fg=TEXT,font=('Segoe UI',11))
        options.update(kwargs)
        return tk.Label(parent,text=text,**options)

    def card(self,parent,bg=PANEL):
        return tk.Frame(parent,bg=bg,highlightbackground=BORDER,highlightthickness=1)

    def build_ui(self):
        # A quiet editorial palette: warm paper, forest ink, and citron accents.
        header=tk.Frame(self.root,bg=BG); header.pack(fill='x',padx=28,pady=(16,12))
        mark=tk.Canvas(header,width=48,height=48,bg=BG,highlightthickness=0)
        mark.pack(side='left',padx=(0,12))
        mark.create_oval(1,1,47,47,fill=DARK,outline=DARK)
        mark.create_text(24,24,text='s',font=('Georgia',29,'italic'),fill=ACCENT)
        brand=tk.Frame(header,bg=BG); brand.pack(side='left')
        self.label(brand,'sinabtanay',font=('Segoe UI',23,'bold')).pack(anchor='w')
        self.label(brand,'A little movement. A lot to say.',fg=MUTED,font=('Segoe UI',10)).pack(anchor='w')
        self.label(header,'  ON YOUR DEVICE  ',bg='#E4EADB',fg=DARK,font=('Segoe UI',9,'bold'),padx=12,pady=9).pack(side='right')
        intro=tk.Frame(self.root,bg=BG); intro.pack(fill='x',padx=30,pady=(0,12))
        self.label(intro,'Make yourself understood.',font=('Georgia',27)).pack(anchor='w')
        self.label(intro,'Shape a letter, make it yours, and give your words a voice.',fg=MUTED,font=('Segoe UI',11)).pack(anchor='w',pady=(6,0))
        body=tk.Frame(self.root,bg=BG); body.pack(fill='both',expand=True,padx=28)
        body.columnconfigure(0,weight=3,minsize=535); body.columnconfigure(1,weight=2,minsize=390)
        body.rowconfigure(0,weight=1)
        camera_panel=self.card(body); camera_panel.grid(row=0,column=0,sticky='nsew',padx=(0,18))
        camera_header=tk.Frame(camera_panel,bg=PANEL); camera_header.pack(fill='x',padx=20,pady=17)
        self.label(camera_header,'01  /  YOUR SIGN',fg=MUTED,font=('Segoe UI',9,'bold')).pack(side='left')
        self.label(camera_header,'● LIVE',fg='#51845D',font=('Segoe UI',9,'bold')).pack(side='right')
        preview_frame=tk.Frame(camera_panel,bg=DARK); preview_frame.pack(fill='both',expand=True,padx=16)
        self.preview=self.label(preview_frame,'Your camera is getting ready…',bg=DARK,fg='#CEDCCF')
        self.preview.pack(fill='both',expand=True)
        camera_footer=tk.Frame(camera_panel,bg=PANEL); camera_footer.pack(fill='x',padx=20,pady=18)
        self.hand_status=self.label(camera_footer,'Hands detected: --',font=('Segoe UI',10,'bold')); self.hand_status.pack(anchor='w')
        self.label(camera_footer,'One hand in view. A comfortable pose. Your own pace.',fg=MUTED,font=('Segoe UI',10),wraplength=480,justify='left').pack(anchor='w',pady=(6,0))
        side=tk.Frame(body,bg=BG); side.grid(row=0,column=1,sticky='nsew')
        detection=self.card(side,DARK); detection.pack(fill='x',pady=(0,14))
        detection_top=tk.Frame(detection,bg=DARK); detection_top.pack(fill='x',padx=20,pady=(16,4))
        self.label(detection_top,'02  /  THE LETTER',fg='#BACBBC',font=('Segoe UI',9,'bold')).pack(anchor='w')
        result_row=tk.Frame(detection,bg=DARK); result_row.pack(fill='x',padx=20)
        self.letter=self.label(result_row,'--',fg=ACCENT,font=('Segoe UI',48,'bold')); self.letter.pack(side='left',padx=(0,18))
        result_details=tk.Frame(result_row,bg=DARK); result_details.pack(side='left',fill='x',expand=True)
        self.recognition_status=self.label(result_details,'Waiting for a hand',fg='#FFFFFF',font=('Segoe UI',11),wraplength=235,justify='left'); self.recognition_status.pack(anchor='w')
        self.score=self.label(result_details,'Model score: --',fg='#BACBBC',font=('Segoe UI',10)); self.score.pack(anchor='w',pady=(6,0))
        style=ttk.Style(self.root); style.theme_use('clam')
        style.configure('Score.Horizontal.TProgressbar',troughcolor='#30483D',background=ACCENT,bordercolor=DARK,lightcolor=ACCENT,darkcolor=ACCENT)
        self.progress=ttk.Progressbar(detection,maximum=100,style='Score.Horizontal.TProgressbar'); self.progress.pack(fill='x',padx=20,pady=(4,18),ipady=0)
        composer=self.card(side); composer.pack(fill='both',expand=True)
        composer_header=tk.Frame(composer,bg=PANEL); composer_header.pack(fill='x',padx=20,pady=(16,8))
        self.label(composer_header,'03  /  YOUR WORDS',fg=MUTED,font=('Segoe UI',9,'bold')).pack(side='left')
        self.count=self.label(composer_header,'0 / 200 characters',fg=MUTED,font=('Segoe UI',9)); self.count.pack(side='right')
        self.message_view=tk.Text(composer,height=2,width=1,wrap='word',font=('Georgia',20),bg='#F8F9F4',fg=TEXT,relief='flat',highlightthickness=0,padx=14,pady=12,takefocus=False,state='disabled')
        self.message_view.pack(fill='both',expand=True,padx=18,pady=(0,12))
        self.buttons={}
        def button(parent,name,key,text,bg,fg,**grid):
            b=tk.Button(parent,text=text,command=lambda:self.action(key),font=('Segoe UI',10,'bold'),bg=bg,fg=fg,activebackground='#CEDFAD' if bg==ACCENT else '#E9EDE3',activeforeground=TEXT,disabledforeground='#9CA69A',relief='flat',borderwidth=0,padx=12,pady=8,takefocus=False,cursor='hand2')
            b.grid(**grid); self.buttons[name]=b
        primary=tk.Frame(composer,bg=PANEL); primary.pack(fill='x',padx=18)
        primary.columnconfigure(0,weight=1)
        button(primary,'Add letter',13,'＋  Add letter     ↵',ACCENT,DARK,row=0,column=0,sticky='ew')
        editing=tk.Frame(composer,bg=PANEL); editing.pack(fill='x',padx=18,pady=8)
        for i,(name,key,text) in enumerate([('Space',32,'Space'),('Delete',8,'⌫  Delete'),('Clear',ord('c'),'Clear')]):
            editing.columnconfigure(i,weight=1)
            button(editing,name,key,text,'#EEF1E8',TEXT,row=0,column=i,sticky='ew',padx=(0,6) if i<2 else 0)
        voice=tk.Frame(composer,bg=PANEL); voice.pack(fill='x',padx=18,pady=(0,6))
        voice.columnconfigure(0,weight=2); voice.columnconfigure(1,weight=1)
        button(voice,'Speak',ord('t'),'Speak my words   →',DARK,'#FFFFFF',row=0,column=0,sticky='ew',padx=(0,6))
        button(voice,'Stop speech',27,'Stop', '#F8E8E1','#854F3C',row=0,column=1,sticky='ew')
        self.buttons['Add letter'].configure(state='disabled')
        if not self.speech.enabled: self.buttons['Speak'].configure(state='disabled')
        self.feedback=self.label(composer,'Wait for a stable letter, then add it.',fg=MUTED,font=('Segoe UI',9),wraplength=350,justify='left'); self.feedback.pack(anchor='w',padx=20,pady=(6,3))
        self.speech_status=self.label(composer,self.speech.status,fg=MUTED,font=('Segoe UI',9),wraplength=350,justify='left'); self.speech_status.pack(anchor='w',padx=20,pady=(0,14))
        footer=tk.Frame(self.root,bg=BG); footer.pack(fill='x',padx=30,pady=(10,12))
        self.label(footer,'ENTER add   /   SPACE separate   /   ⌫ delete   /   C clear   /   T speak   /   ESC stop   /   Q exit',fg=MUTED,font=('Segoe UI',9)).pack(anchor='w')
        self.label(footer,'24 static letters · J/Z excluded · Model scores are not accuracy · Messages stay in memory',fg=MUTED,font=('Segoe UI',9)).pack(anchor='w',pady=(5,0))

    def keypress(self,event):
        special={'Return':13,'KP_Enter':13,'space':32,'BackSpace':8,'Escape':27}
        key=special.get(event.keysym)
        if key is None and event.char: key=ord(event.char.lower())
        if key in (ord('q'),ord('Q')): self.close()
        elif key is not None: self.action(key)
        return 'break'

    def action(self,key):
        if key in (ord('t'),ord('T')): feedback=self.speech.speak(self.message.text)
        elif key==27: feedback=self.speech.stop()
        else: feedback=self.message.handle_key(key,self.current_label)
        if feedback is None: return
        self.feedback.configure(text=feedback)
        self.message_view.configure(state='normal')
        self.message_view.delete('1.0','end'); self.message_view.insert('1.0',self.message.text)
        self.message_view.configure(state='disabled'); self.message_view.see('end')
        self.count.configure(text=f'{len(self.message.text)} / {self.message.limit} characters')

    def tick(self):
        if self.closed: return
        self.after_id=None
        try:
            ok,frame=self.camera.read()
            if not ok or frame is None or not frame.size: raise RuntimeError('Camera frame unavailable. Close and reopen the app.')
            if settings.MIRROR_CAMERA: frame=cv2.flip(frame,1)
            timestamp=max(time.monotonic_ns()//1_000_000,self.last_timestamp+1); self.last_timestamp=timestamp
            result=self.detector.detect(frame,timestamp)
            self.current_label=None; confidence=None
            if not result.hand_landmarks:
                self.predictor.reset(); status='No hand detected'
            else:
                h,w=frame.shape[:2]
                try: features=create_feature_vector(result,(w,h))
                except ValueError:
                    self.predictor.reset(); status='Ambiguous hand — reposition'
                else: self.current_label,confidence,status=self.predictor.predict(features)
            self.letter.configure(text=self.current_label or '--')
            self.recognition_status.configure(text=status)
            self.score.configure(text=f'Model score: {confidence:.0%}' if confidence is not None else 'Model score: --')
            self.progress['value']=100*confidence if confidence is not None else 0
            self.buttons['Add letter'].configure(state='normal' if self.current_label else 'disabled')
            self.hand_status.configure(text=f'Hands detected: {len(result.hand_landmarks)}')
            draw_hand_results(frame,result)
            h,w=frame.shape[:2]
            max_w=min(640,max(100,self.preview.winfo_width()-4)); max_h=min(480,max(100,self.preview.winfo_height()-4))
            scale=min(max_w/w,max_h/h)
            frame=cv2.resize(frame,(max(1,int(w*scale)),max(1,int(h*scale))))
            ok,ppm=cv2.imencode('.ppm',frame)
            if not ok: raise RuntimeError('Cannot render camera preview.')
            self.photo=tk.PhotoImage(master=self.root,data=ppm.tobytes(),format='PPM')
            self.preview.configure(image=self.photo,text='')
        except (OSError,ValueError,RuntimeError,cv2.error) as error:
            self.current_label=None; self.predictor.reset()
            self.letter.configure(text='--'); self.progress['value']=0
            self.score.configure(text='Model score: --')
            self.hand_status.configure(text='Hands detected: --')
            self.preview.configure(image='',text='Camera unavailable')
            self.buttons['Add letter'].configure(state='disabled')
            self.recognition_status.configure(text=str(error))
            self.release_camera()
            self.poll_speech()
            return
        self.speech_status.configure(text=self.speech.poll())
        self.after_id=self.root.after(30,self.tick)

    def poll_speech(self):
        if not self.closed:
            self.speech_status.configure(text=self.speech.poll())
            self.after_id=self.root.after(100,self.poll_speech)

    def release_camera(self):
        if self.camera is not None: self.camera.release(); self.camera=None
        if self.detector is not None: self.detector.close(); self.detector=None

    def close(self):
        if self.closed: return
        self.closed=True
        if self.after_id is not None: self.root.after_cancel(self.after_id)
        try: self.release_camera()
        finally:
            self.speech.close()
            self.root.destroy()


# Compatibility for existing integrations; the application brand is Sinabtanay.
SignBridgeApp = SinabtanayApp
