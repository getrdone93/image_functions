import cv2
import tkinter as tk
from PIL import ImageTk, Image
import time

def build_window(title):
    w_dim = 800
    h_dim = 800

    window = tk.Tk()
    window.title(title)
    window.geometry('800x800')
    window.configure(background='grey')
    
    return window

def build_gui():

    w1 = build_window(title='Example 1')
    w2 = build_window(title='Example 2')
    w3 = build_window(title='Example 3')

    ws = (w1, w2, w3)

    while True:
        for w in ws:
            w.update_idletasks()
            w.update()

        time.sleep(0.5)

if __name__ == '__main__':
    build_gui()
