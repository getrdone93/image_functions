import numpy as np
import math

def convolution(padded_segment, kernel, kc, ds, de, p):
    data_segment = padded_segment[ds:de, ds:de]
    new_seg = np.zeros(data_segment.shape)
    for ri, rv in enumerate(data_segment):
        for ci, cv in enumerate(data_segment[ri]):
            im_seg = segment_portion(segment=padded_segment, x=ri + p, y=ci + p, kc=kc)
            new_seg[ri][ci] = np.sum(im_seg * kernel)
    return new_seg

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

def log_operator(sigma, const_factor):
    m = math.ceil(sigma * 7)
    center = m // 2
    result = np.zeros((m, m))
    for ri, rv in enumerate(result):
        for ci, cv in enumerate(result[ri]):
            r_sq = ri**2 + ci**2
            sig_sq = sigma**2
            result[ri][ci] = ((r_sq - sig_sq) / sig_sq) - (r_sq / 2 * sig_sq)
    return result
    

if __name__ == '__main__':
    print(log_operator(sigma=1.0, const_factor=50))
