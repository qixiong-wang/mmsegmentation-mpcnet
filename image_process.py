import cv2
import numpy as np
import matplotlib.pyplot as plt
def convert_to_color(arr_2d, palette):
    """ Numeric labels to RGB-color encoding """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d

def convert_from_color(arr_3d, palette):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        c=c[::-1]
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i
    return arr_2d

palette = \
    {
        0: (0, 0, 0),  ##bg
        1: (0, 0, 63),  ##ship
        2: (0, 63, 63),
        3: (0, 63, 0),
        4: (0, 63, 127),
        5: (0, 63, 191),
        6: (0, 63, 255),
        7: (0, 127, 63),
        8: (0, 127, 127),
        9: (0, 0, 127),  ## small vehicle
        10: (0, 0, 191),
        11: (0, 0, 255),
        12: (0, 191, 127),
        13: (0, 127, 191),
        14: (0, 127, 255),  ## plane
        15: (0, 100, 155)   ## harbor
    }

# palette = {0 : (255, 255, 255), # Impervious surfaces (white)
#            1 : (0, 0, 255),     # Buildings (blue)
#            2 : (0, 255, 255),   # Low vegetation (cyan)
#            3 : (0, 255, 0),     # Trees (green)
#            4 : (255, 255, 0),   # Cars (yellow)
#            5 : (255, 0, 0),     # Clutter (red)
#            6 : (199,196,196),
#            255: (255, 255, 0)}       # Undefined (black)

# invert_palette = {(255, 255, 255) : 0, # Impervious surfaces (white)
#                    (0, 0, 255) : 1,     # Buildings (blue)
#                    (0, 255, 255): 2 ,   # Low vegetation (cyan)
#                    (0, 255, 0) : 3,     # Trees (green)
#                    (255, 255, 0) : 4,   # Cars (yellow)
#                    (255, 0, 0) : 5,     # Clutter (red)
#                    (0,0,0) : 6}       # Undefined (black)




invert_palette = {v: k for k, v in palette.items()}
# img = cv2.imread('./Vaihingen/ISPRS_semantic_labeling_Vaihingen/gts_for_participants/top_mosaic_09cm_area32.tif')[0:450,680:1100,::-1
# img = cv2.imread('./P0777.png')
# gt = cv2.imread('./P0777_instance_color_RGB.png')
img = cv2.imread('./image299_1_001.png')
gt = cv2.imread('./image299_1_002.png')
gt_label = convert_from_color(gt,invert_palette)

target_color = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)

pixels = target_color == 0

target_color[pixels[:,:,0]] = np.array([155,100,0])



img[np.where(gt_label==1)] = img[np.where(gt_label==1)]*0.5+target_color[np.where(gt_label==1)]*0.5
# img[np.where(np.sum(gt,axis=2)!=0)] = img[np.where(np.sum(gt,axis=2)!=0)]*0.5+gt[np.where(np.sum(gt,axis=2)!=0)]*0.5
# img = img*0.5+gt*0.5

# img = cv2.imread('./1.png')[:,:,::-1]

# print(img.shape)
# gt = convert_from_color(gt,invert_palette)


# for i in range(6):
#     new_img = np.ones_like(img) * 6
#     new_img[img==i] = i
#     new_img = convert_to_color(new_img,palette)[:,:,::-1]
#     cv2.imwrite('./img_{}.png'.format(i),new_img)
# print(img.shape)

# img[np.where(contour==255)][1] = np.array([255])
# img[np.where(contour==255)][2] = np.array([255])
cv2.imwrite('./image299_1_merged_target.png',img)


# img = cv2.imread('./Vaihingen/ISPRS_semantic_labeling_Vaihingen/gts_for_participants/top_mosaic_09cm_area32.tif')[0:450,680:1100,:]
# cv2.imwrite('./img_1.png', img)
# img = cv2.imread('./Vaihingen/ISPRS_semantic_labeling_Vaihingen/top/top_mosaic_09cm_area32.tif')[0:450,680:1100,:]
# cv2.imwrite('./img_raw.png', img)