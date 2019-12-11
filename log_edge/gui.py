import numpy as np
import math
import cv2
import tkinter as tk
from PIL import ImageTk, Image

def pad_segment(segment, kcy):
    h = len(segment)
    pad = kcy * 2
    ns = np.add(segment.shape, pad)
    padded = np.zeros(np.product(ns)).reshape(ns)
    ds, de = (pad // 2, ns[0] - (pad // 2))
    padded[ds:de, ds:de] = segment
    return padded, (ds, de), pad

def segment_portion(segment, x, y, kc):
    bx, ex = (x - kc, x + kc + 1)
    by, ey = (y - kc, y + kc + 1)
    return segment[by:ey, bx:ex]

def convolution(segment, kernel, kc, ds, de, p):
    data_segment = segment[ds:de, ds:de]
    new_seg = np.zeros(data_segment.shape)
    for ri, rv in enumerate(data_segment):
        for ci, cv in enumerate(data_segment[ri]):
            im_seg = segment_portion(segment=segment, x=ri + p, y=ci + p, kc=kc)
            if im_seg.shape != kernel.shape:
                filler = np.zeros((kernel.shape))
                filler[:im_seg.shape[0], :im_seg.shape[1]] = im_seg
                im_seg = filler
            new_seg[ri][ci] = np.sum(im_seg * kernel)
    return new_seg

def sign(num):
    return num >= 0

def zero_cross(segment, x, y, s_func=sign):
    left = s_func(num=segment[x-1][y])
    right = s_func(num=segment[x+1][y])
    up = s_func(num=segment[x][y-1])
    down = s_func(num=segment[x][y+1])
    up_left = s_func(num=segment[x-1][y-1])
    down_right = s_func(num=segment[x+1][y+1])
    up_right = s_func(num=segment[x+1][y-1])
    down_left = s_func(num=segment[x-1][y+1])
    return (left != right) or (up != down) or (up_left != down_right) or (up_right != down_left)

def log_kernel_size(sigma):
    m = math.ceil(sigma * 7)
    dim = len(range(-m, m + 1))
    return dim, dim // 2

def x_y_vals(m, center):
    grid = np.zeros((m, m)).tolist()
    sx = -center
    sy = -center
    for ri, rv in enumerate(grid):
        for ci, cv in enumerate(grid[ri]):
            grid[ri][ci] = (sx + ri, sy + ci)
    return grid

def log_kernel(sigma, const_factor, m, xy_grid):
    result = np.zeros((m, m))
    for ri, rv in enumerate(result):
        for ci, cv in enumerate(result[ri]):
            x, y = xy_grid[ri][ci]
            r_sq = x**2 + y**2
            sig_sq = sigma**2
            sig_cube = sigma**4
            result[ri][ci] = (((r_sq - sig_sq) / sig_sq) - (r_sq / 2 * sig_sq)) + const_factor

    result = np.asarray([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    #result = np.asarray([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    return result

def apply_image(image, func, func_args):
    result = []
    for d in range(image.shape[2]):
        result.append(func(segment=image[:, :, d], **func_args))
    return result

def zero_crossing(segment, ds, de, zc_func=zero_cross):
    data_segment = segment[ds:de + 1, ds:de + 1]
    crossing_segment = np.zeros(data_segment.shape)
    for ri, rv in enumerate(data_segment):
        for ci, cv in enumerate(data_segment[ri]):
            r, g, b = segment[ri, ci, :]
            zc = all(map(lambda d: zero_cross(segment=segment[:, :, d], x=ri, y=ci), 
                    (0, 1, 2)))
            if 0 in {r, g, b} and zc:
                crossing_segment[ri][ci][:] = 255
            else:
                crossing_segment[ri][ci][:] = 0
    return crossing_segment

def log_image(sigma, image):
    m, center = log_kernel_size(sigma=sigma)
    grid = x_y_vals(m=m, center=center)
    kernel = log_kernel(sigma=sigma, const_factor=0, m=m, xy_grid=grid)
    padded = apply_image(image=image, func=pad_segment, func_args={'kcy': center})
    pi_sh, dse, pad = padded[0]
    new_shape = (pi_sh.shape[0], pi_sh.shape[1], image.shape[-1])
    ds, de = dse
    ps = np.transpose(np.asarray(list(map(lambda t: t[0], padded))), (1, 2, 0))

    log_image = apply_image(image=ps, func=convolution, func_args={'kernel': kernel, 'kc': center, 
                                                                 'ds': ds, 'de': de, 'p': pad})
    log_tensor = np.transpose(np.asarray(log_image), (1, 2, 0))
    zero_cross = zero_crossing(segment=log_tensor, ds=ds, de=de)
    zero_cross_tensor = np.transpose(np.asarray(zero_cross), (1, 0, 2))
    return zero_cross_tensor

def test_log():
    kobe = cv2.resize(cv2.imread('./images/kobe_bryant.jpg'), (300, 300))
    log_kobe = log_image(sigma=0.1, image=kobe)
    print(log_kobe.shape)
    cv2.imwrite('./log_kobe.jpg', log_kobe)

def build_gui(image_path, log_image_path):
    start_y = 50
    image_gap = 40
    w_dim = 800
    h_dim = 300
    label_gap = 20
    slider_gap = 5
    button_gap = 60
    image_label_gap = 20
    image_shape = (224, 224)

    window = tk.Tk()
    window.title('LoG Eddge Detection')
    window.geometry('800x300')
    window.configure(background='grey')

    #images
    raw_image = Image.fromarray(cv2.cvtColor(cv2.resize(
        cv2.imread(image_path), image_shape), cv2.COLOR_BGR2RGB), 'RGB')
    image = ImageTk.PhotoImage(raw_image)

    raw_log_image = Image.fromarray(cv2.cvtColor(cv2.resize(
        cv2.imread(log_image_path), image_shape), cv2.COLOR_BGR2RGB), 'RGB')
    log_image = ImageTk.PhotoImage(raw_log_image)

    #left side
    og  = 'Original'
    panel1 = tk.Label(image=image)
    panel1.place(x=0, y=start_y)

    #right side
    edge_str = 'Edge Image'
    panel2 = tk.Label(image=log_image)
    panel2.place(x=w_dim - log_image.width(), y=start_y)
    slider_x = tk.Scale(window, from_=0.1, to=5.0, digits = 2, resolution=0.1, 
                        orient=tk.HORIZONTAL, variable=0, length=w_dim-10)
    slider_x.set(0)
    slider_x.place(x=0, y=slider_gap)

    #LoG button
    lb_w, lb_h = 5, 1
    log_button = tk.Button(window, text='LoG', width=lb_w, height=lb_h)
    log_button.place(x=(w_dim / 2) - lb_w, y=(h_dim / 2) - lb_h / 2)

    window.mainloop()

if __name__ == '__main__':
    image_path = './images/kobe_bryant.jpg'
    log_image_path = './images/log_kobe.jpg'
    build_gui(image_path=image_path, log_image_path=log_image_path)
