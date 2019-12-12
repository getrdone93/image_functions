import cv2
import log_funcs
import os.path as path
import numpy as np
import math
import operator as op
import matplotlib.pyplot as plt

BASE_DIR = './images/original'
LOG_DIR = './images/LoG'
PSPACE_DIR = './images/pspace'

EXAMPLE_ARGS = {'Example_1': {'original_image': path.join(BASE_DIR, 'kobe_bryant.jpg'), 
                              'log_image': path.join(LOG_DIR, 'log_kb.jpg'), 
                              'pspace_image': path.join(PSPACE_DIR, 'kb.png')},
                   'Example_2': {'original_image': path.join(BASE_DIR, 'field.jpg'), 
                                 'log_image': path.join(LOG_DIR, 'log_field.jpg'), 
                                 'pspace_image': path.join(PSPACE_DIR, 'lf.png')}, 
                   'Example_3': {'original_image': path.join(BASE_DIR, 'flower.jpg'), 
                                 'log_image': path.join(LOG_DIR, 'log_flower.jpg'), 
                                 'pspace_image': path.join(PSPACE_DIR, 'flower.png')}}

def compute_logs(image_paths, example_args, image_shape):
    for ip, op, ex in image_paths:
        print("Computing LoG for {}".format(ip))
        image = cv2.resize(cv2.imread(ip), image_shape)
        log_image = cv2.resize(log_funcs.log_image(sigma=1, image=image), image_shape)
        example_args[ex].update({'log_image': log_image})

def compute_all_logs(example_args):
    # image_paths = ((path.join(BASE_DIR, 'kobe_bryant.jpg'), path.join(LOG_DIR, 'log_kb.jpg'), 'Example_1'),
    #                (path.join(BASE_DIR, 'flower.jpg'), path.join(LOG_DIR, 'log_flower.jpg'), 'Example_2'), 
    #                 (path.join(BASE_DIR, 'field.jpg'), path.join(LOG_DIR, 'log_field.jpg'), 'Example_3'))
    image_paths = ((path.join(BASE_DIR, 'kobe_bryant.jpg'), path.join(LOG_DIR, 'log_kb.jpg'), 'Example_1'),)
    compute_logs(image_paths=image_paths, example_args=example_args, image_shape=(300, 300))

def p_space(points):
    pm = {}
    thetas = range(0, 181)
    for x, y in points:
        theta_map = {}
        for t in thetas:
            theta = round(math.radians(t))
            p = x * math.cos(theta) + y * math.sin(theta)
            if (theta, p) not in theta_map:
                theta_map[(theta, p)] = 0
            theta_map[(theta, p)] += 1 
        pm[(x, y)] = theta_map
        
    return pm, thetas

def local_maxes(p_space):
    result = {}
    for k, v in p_space.items():
        result[k] = max(v.items(), key=op.itemgetter(1))
    return result

def edge_points(image):
    x, y, _ = image.shape
    ep = []
    for r in range(0, x):
        for c in range(0, y):
            d1, d2, d3 = map(lambda d: int(d), image[r, c, :])
            if {d1, d2, d3} == {255}:
                ep.append((r, c))
    return ep

def plot_p_space(space, path):
    thetas, ps = [], []
    for point, tp in space.items():
        th_p, _ = tp
        theta, p = th_p
        xs = []
        ys = []
        thetas.append(theta)
        ps.append(p)

    plt.scatter(thetas, ps)

    plt.xlabel('theta')
    plt.ylabel('p')
    plt.legend(loc='best')
    plt.savefig(path)

def compute_p_space(log_image):
    ep = edge_points(image=log_image)
    ps, _ = p_space(points=ep)
    lm = local_maxes(p_space=ps)
    return lm

def show_examples(example_args, image_shape):
    temp_args = {k: example_args[k] for k in {'Example_1'}}
    for ex, ex_args in temp_args.items():
        oi_image = cv2.resize(cv2.imread(ex_args['original_image']), image_shape)
        log_image = ex_args['log_image']
        lm = compute_p_space(log_image=log_image)
        plot_p_space(space=lm, path=ex_args['pspace_image'])
        pspace_image = cv2.imread(ex_args['pspace_image'], cv2.IMREAD_UNCHANGED)

        show_images = ((ex + '_original', oi_image), (ex + '_LoG', log_image), 
                       (ex + '_pspace', pspace_image))
        for title, img in show_images:
            cv2.imshow(title, img)

    cv2.waitKey(0)

if __name__ == '__main__':
    compute_all_logs(example_args=EXAMPLE_ARGS)
    show_examples(example_args=EXAMPLE_ARGS, image_shape=(300, 300))
