import gui
import cv2
import os.path as path

BASE_PATH = './images'

def examples(image_paths):
    for ip, ob, _ in image_paths:
        print("Computing LoG for {}...".format(ip))
        image = cv2.resize(cv2.imread(ip), (300, 300))
        log_img = gui.log_image(sigma=1, image=image)
        
        cv2.imshow('original_' + ob, image)
        cv2.imshow('LoG_' + ob, log_img)


    stars = 100
    for _, ob, descr in image_paths:
        print()
        print("*" * stars)
        print("{} description: {}".format(ob.capitalize(), descr))
        print("*" * stars)
        print()

    print("\n\tClick on one of the windows and hit ENTER to exit the program.\n")
    cv2.waitKey(0)

if __name__ == '__main__':
    flower_descr = "The flower is a good example for LoG because the lines are distinguishable\n"\
                   + "from background. The pedals serve as a boundary as the intensities are\n"\
                   + "different from the green background. This helps the LoG kernel find the\n"\
                   + "edges."
    tree_descr = "The tree is a bad example for edge detection. This image makes it difficult\n"\
                 + "for the LoG kernel because the intensities are nearly all the same. Also,\n"\
                 + "all off the lines are vertical which means a horizontal kernel would not work\n"\
                 + "well on this image input."
    field_descr = "The field is a mediocre example of LoG edge detection. The image contains almost\n"\
                  "no edges. The sky and field are not good inputs because they do not have many edges. The\n"\
                  "LoG kernel can find the horizon line, but it is difficult to see. It also picks up\n"\
                  "some of the edges of the clouds, but these are difficult to see as well."
    image_paths = ((path.join(BASE_PATH, 'flower.jpg'), 'flower', flower_descr),
                   (path.join(BASE_PATH, 'tree.jpg'), 'tree', tree_descr),
                   (path.join(BASE_PATH, 'field.jpg'), 'field', field_descr))
    examples(image_paths=image_paths)
