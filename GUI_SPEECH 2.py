from tkinter import *
from tkinter import ttk
import xlsxwriter
import xlrd
from tkinter import filedialog
import numpy as np
import numpy
import matplotlib.pyplot as plt
import cv2
from scipy.stats import skew,kurtosis
from DNN_TRN import Calc_Wt2
import pywt


def build_filters():
 filters = []
 ksize = 31
 for theta in np.arange(0, np.pi, np.pi / 16):
     kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
     kern /= 1.5*kern.sum()
     filters.append(kern)
 return filters
 
def process(img, filters):
     accum = np.zeros_like(img)
     for kern in filters:
         fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
         np.maximum(accum, fimg, accum)
     return accum
def thresholded(center, pixels):
    out = []
    for a in pixels:
        if a >= center:
            out.append(1)
        else:
            out.append(0)
    return out

def get_pixel_else_0(l, idx, idy, default=0):
    try:
        return l[idx,idy]
    except IndexError:
        return default



workbook = xlsxwriter.Workbook('demo.xlsx')
worksheet = workbook.add_worksheet()

#worksheet.set_column('A:A', 20)
bold = workbook.add_format({'bold': True})
worksheet.write('A1', 'USERNAME')
worksheet.write('B1', 'PASSWORD')
worksheet.write('C1', 'MOBILE NUMBER')
worksheet.write('D1', 'ROLL NUMBER')
worksheet.write('E1', 'EMAIL ID')

window = Tk()
window.title("CBIR")
window.geometry('1200x500')

tab_control = ttk.Notebook(window)
tab1 = ttk.Frame(tab_control)
tab2 = ttk.Frame(tab_control)
tab3 = ttk.Frame(tab_control)
tab_control.add(tab1, text='STUDENT REGISTRATION')
tab_control.add(tab2, text='CBIR')

#############################################################################################################################################################
# HEADING
def show_entry_fields():
   print("First Name: %s\nLast Name: %s" % (e1.get(), e2.get()))
   Un=e1.get()
   Pw=e2.get()
   print((Un))
   res = "PERSON " + Un + " IS ADDED"
   lbl1.configure(text= res)
   worksheet.write(str('A'+ str(2)),str(Un) )
   worksheet.write(str('B'+ str(2)),str(Pw) )
   workbook.close()

def TST_SPEECH():
   Un=ee1.get()
   Pw=ee2.get()
   print('USERNAME',Un)
   wb = xlrd.open_workbook('demo.xlsx') 
   sheet = wb.sheet_by_index(0) 
   Un1=sheet.cell_value(1, 0)
   Pw1=sheet.cell_value(1, 1)
   print('UN',Un1);
   print('PW',Pw1);
   if Un==Un1 and Pw==Pw1:
      messagebox.showinfo('LOGIN SUCCESSFUL', 'WELCOME')
      TRR=[1,1,1,1,1];
      np.savetxt('aa.txt',TRR)
      lbl1.configure(text='SUCCESSFUL LOGIN')
   else:
      TRR=[2,1,1,1,1];
      np.savetxt('aa.txt',TRR)
      lbl1.configure(text=' ACCESS DENIED')
      messagebox.showerror('LOGIN DENIED', 'Wrong Username Or Password')
      window.quit()
      window.destroy()

def RST():
      messagebox.showerror('LOGIN DENIED', 'Wrong Username Or Password')
      window.quit()
      window.destroy()

   
def GEN_SPEECH():
    KK=np.loadtxt('aa.txt')
    if KK[0]==1:
         fname1 = filedialog.askopenfilename(filetypes = (("Template files", "*.jpg"), ("All files", "*")))
         I=cv2.imread(fname1)
         #2:PREPROCESS
         # RESIZING
         I= cv2.resize(I,(32,32),0)
         OI=I;
         #3:ENHANCEMENT
         # MEDIAN FILTER
         I = cv2.medianBlur(I,5)
         # FEATURE EXTRACTION
         # DWT
         I=I[:,:,0]
         coeffs2 = pywt.dwt2(I, 'bior1.3')
         LL, (LH, HL, HH) = coeffs2
         
         FV1=np.median(LL) 
         FV2=np.mean(LL)
         FV3=skew(skew(LL))
         FV4=kurtosis(kurtosis(LL))
         
         FV5=np.median(LH) 
         FV6=np.mean(LH)
         FV7=skew(skew(LH))
         FV8=kurtosis(kurtosis(LH))
         
         FV9=np.median(HL) 
         FV10=np.mean(HL)
         FV11=skew(skew(HL))
         FV12=kurtosis(kurtosis(HL))
         
         FV13=np.median(HH) 
         FV14=np.mean(HH)
         FV15=skew(skew(HH))
         FV16=kurtosis(kurtosis(HH))
         
         TST=np.array([FV1,FV2,FV3,FV4,FV5,FV6,FV7,FV8,FV9,FV10,FV11,FV12,FV13,FV14,FV15,FV16])
         TST=np.transpose(TST)
         print(TST)
         print(type(TST))
         TRR=np.loadtxt('a.txt')
         IND,IND2=Calc_Wt2(TST)
         print('RESULT',IND)
         print('RESULT',IND2)
         IND2=np.round(IND2)
         img1=OI
         plt.figure()
         plt.subplot(231),plt.imshow(OI,'gray'),plt.title('ORIGINAL')
         filename=('TEST/1 ('+str(IND2[0]+1)+').jpg');print(filename); img1=cv2.imread(filename)
         plt.subplot(232),plt.imshow(img1,'gray'),plt.title('ONE')
         filename=('TEST/1 ('+str(IND2[1]+1)+').jpg'); img1=cv2.imread(filename)
         plt.subplot(233),plt.imshow(img1,'gray'),plt.title('TWO')
         filename=('TEST/1 ('+str(IND2[2]+1)+').jpg'); img1=cv2.imread(filename)
         plt.subplot(234),plt.imshow(img1,'gray'),plt.title('THREE')
         filename=('TEST/1 ('+str(IND2[3]+1)+').jpg'); img1=cv2.imread(filename)
         plt.subplot(235),plt.imshow(img1,'gray'),plt.title('FOUR')
         filename=('TEST/1 ('+str(IND2[4]+1)+').jpg'); img1=cv2.imread(filename)
         plt.subplot(236),plt.imshow(img1,'gray'),plt.title('FIVE')
         plt.show()


    else:
       lbl1.configure(text='ACCESS DENIED')
          

                                            
      
#######################################################################################################
lbl = Label(tab1, text="                     ",font=("Arial Bold", 30),foreground =("red"))
lbl.grid(column=0, row=0)
lbl = Label(tab1, text="STUDENT  REGISTRATION  DETAILS",font=("Arial Bold", 30),foreground =("red"),background  =("black"))
lbl.grid(column=1, row=0)
lbl = Label(tab1, text="                  ",font=("Arial Bold", 30),foreground =("red"))
lbl.grid(column=2, row=0)
# USERNAME & PASSWORD ENTRY BOX
Label(tab1, text="USERNAME",font=("Arial Bold", 15),foreground =("green")).grid(row=1,column=0)
Label(tab1, text="PASSWORD",font=("Arial Bold", 15),foreground =("green")).grid(row=2,column=0)
Label(tab1, text="MOBILE NUMBER",font=("Arial Bold", 15),foreground =("green")).grid(row=3,column=0)
Label(tab1, text="ROLL NUMBER",font=("Arial Bold", 15),foreground =("green")).grid(row=4,column=0)
Label(tab1, text="EMAIL ID",font=("Arial Bold", 15),foreground =("green")).grid(row=5,column=0)
e1 = Entry(tab1)
e2 = Entry(tab1)
e3 = Entry(tab1)
e4 = Entry(tab1)
e5 = Entry(tab1)
e1.grid(row=1, column=1,sticky=W, pady=20)
e2.grid(row=2, column=1,sticky=W, pady=20)
e3.grid(row=3, column=1,sticky=W, pady=20)
e4.grid(row=4, column=1,sticky=W, pady=20)
e5.grid(row=5, column=1,sticky=W, pady=20)

lbl1 = Label(tab1, text="  STATUS   ",font=("Arial Bold", 15),foreground =("red"),background  =("white"))
lbl1.grid(column=2, row=4,sticky=W, pady=20)
Button(tab1, text='CANCEL', command=tab1.quit,font=("Arial Bold", 15),foreground =("yellow"),background  =("brown")).grid(row=3, column=2, sticky=W, pady=20)
Button(tab1, text='REGISTER', command=show_entry_fields,font=("Arial Bold", 15),foreground =("yellow"),background  =("brown")).grid(row=2, column=2, sticky=W, pady=4)

#############################################################################################################################################################
lbl = Label(tab2, text="CBIR",font=("Arial Bold", 30),foreground =("red"),background  =("white"))
lbl.grid(column=0, row=0)
lbl = Label(tab2, text="SYSTEM",font=("Arial Bold", 30),foreground =("red"),background  =("white"))
lbl.grid(column=1, row=0)
# USERNAME & PASSWORD ENTRY BOX
Label(tab2, text="USERNAME",font=("Arial Bold", 15),foreground =("green")).grid(row=1,column=0,sticky=W, padx=20, pady=20)
Label(tab2, text="PASSWORD",font=("Arial Bold", 15),foreground =("green")).grid(row=2,column=0,sticky=W,  padx=20, pady=20)
ee1 = Entry(tab2)
ee2 = Entry(tab2)
ee1.grid(row=1, column=1,sticky=W, padx=20, pady=20)
ee2.grid(row=2, column=1,sticky=W, padx=20, pady=20)
Button(tab2, text='LOGIN', command=TST_SPEECH,font=("Arial Bold", 15),foreground =("yellow"),background  =("brown")).grid(row=3, column=1, sticky=W, pady=4)
Button(tab2, text='TEST', command=GEN_SPEECH,font=("Arial Bold", 15),foreground =("yellow"),background  =("brown")).grid(row=3, column=2, sticky=W, pady=4)
Button(tab2, text='CANCEL', command=RST,font=("Arial Bold", 15),foreground =("yellow"),background  =("brown")).grid(row=3, column=0, sticky=W, pady=4)
#############################################################################################################################################################
tab_control.pack(expand=1, fill='both')
window.mainloop()
