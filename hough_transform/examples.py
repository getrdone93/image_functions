import cv2
import log_funcs
import os.path as path
import numpy as np
import math
import operator as op
import matplotlib.pyplot as plt

BASE_DIR = './images/original'
PSPACE_DIR = './images/pspace'

EX_1_DESCR = "This is a good example for hough transformation. The basketball player\n"\
             + "has an upright stance which makes for edges to be clearly defined. He\n"\
             + "also stands against a black background which helps the hough transform\n"\
             + "find the vertical lines in the edge image. There are also other player's\n"\
             + "arms that make it in the picture which provide a good input for hough transform."

EX_2_DESCR = "This image is a mediocre example for hough transform. The image has some lines,\n"\
             + "but the vertical lines are all in the clouds and they are not clearly visible.\n"\
             + "The field has no edges and the horizon provides the best chance at hough transform\n"\
             + "finding an edge."

EX_3_DESCR = "The flower is a bad example for hough transformation. The pedals of the flower are\n"\
             + "not straight and the leaves do not provide straight edges for edge detection. The background\n"\
             + "is not as discernable from the leaves which makes it harder to detect edges.\n"\
             + "The stem is also barely visible which does not provide a straight line for hough\n"\
             + "transform to detect."

EXAMPLE_ARGS = {'Example_1': {'original_image': path.join(BASE_DIR, 'kobe_bryant.jpg'), 
                              'log_image': None, 
                              'pspace_image': path.join(PSPACE_DIR, 'kb.png'), 
                              'descr': EX_1_DESCR},
                   'Example_2': {'original_image': path.join(BASE_DIR, 'field.jpg'), 
                                 'log_image': None, 
                                 'pspace_image': path.join(PSPACE_DIR, 'lf.png'), 
                                 'descr': EX_2_DESCR}, 
                   'Example_3': {'original_image': path.join(BASE_DIR, 'flower.jpg'), 
                                 'log_image': None, 
                                 'pspace_image': path.join(PSPACE_DIR, 'flower.png'), 
                                 'descr': EX_3_DESCR}}

def compute_logs(image_paths, example_args, image_shape):
    for ip, ex in image_paths:
        print("Computing LoG for {}".format(ip))
        image = cv2.resize(cv2.imread(ip), image_shape)
        log_image = cv2.resize(log_funcs.log_image(sigma=1, image=image), image_shape)
        example_args[ex].update({'log_image': log_image})

def compute_all_logs(example_args):
    image_paths = ((path.join(BASE_DIR, 'kobe_bryant.jpg'), 'Example_1'),
                   (path.join(BASE_DIR, 'field.jpg'), 'Example_2'), 
                    (path.join(BASE_DIR, 'flower.jpg'), 'Example_3'))
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
    for point, tp in space.items():
        th_p, _ = tp
        theta, p = th_p
        xs = []
        ys = []
        for k, v in space.items():
            x, y = k
            comp_p = x * math.cos(theta) + y * math.sin(theta)
            xs.append((theta, p))
            ys.append(comp_p)

        plt.plot(xs, ys)
            
    plt.xlabel('theta')
    plt.ylabel('p')
    plt.savefig(path)
    plt.clf()

def compute_p_space(log_image):
    ep = edge_points(image=log_image)
    ps, _ = p_space(points=ep)
    lm = local_maxes(p_space=ps)
    return lm

def show_examples(example_args, image_shape):
    stars = 110
    for ex, ex_args in example_args.items():
        oi_image = cv2.resize(cv2.imread(ex_args['original_image']), image_shape)
        log_image = ex_args['log_image']
        lm = compute_p_space(log_image=log_image)
        plot_p_space(space=lm, path=ex_args['pspace_image'])
        pspace_image = cv2.imread(ex_args['pspace_image'], cv2.IMREAD_UNCHANGED)

        show_images = ((ex + '_original', oi_image), (ex + '_LoG', log_image), 
                       (ex + '_pspace', pspace_image), (ex + '_overlay', oi_image))
        for title, img in show_images:
            cv2.imshow(title, img)

        print()
        print("*" * stars)
        print("{} description: \n{}".format(ex, ex_args['descr']))
        print("*" * stars)
        print()

    print("\n\tClick on one of the windows and hit ENTER to exit the program.\n")
    cv2.waitKey(0)

if __name__ == '__main__':
    compute_all_logs(example_args=EXAMPLE_ARGS)
    show_examples(example_args=EXAMPLE_ARGS, image_shape=(300, 300))
