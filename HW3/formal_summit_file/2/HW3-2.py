import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys # used for get intmax

#img_image = cv2.imread('2-image.jpg')
img_image = cv2.imread('2-masterpiece.jpg')

# k-means

# determine points in each cluster
def select_point_to_center(k, point, center): # point, center is array with element = 1x3 array.
    dist_matrix = []
    for i in np.arange(k): 
        # 3d dist_matrix
        dist_matrix.append(point-center[i]) # dist_matrix: kx(# of pixel)x3
    #dist_matrix[i][j][k]: [dist_R, dist_G, dist_B], where i:center[i], img_image[j][k]
    dist_matrix = np.square(dist_matrix)
    dist_matrix = np.sum(dist_matrix, axis=2) # [dist_R, dist_G, dist_B] -sum-> R #size:kx(# of pixel)
    # img_with_cluster_center: size:(# of pixel)(Ximg_image), element:index of center
    img_with_cluster_center = np.argmin(dist_matrix, axis=0) # 2d, axis=0, fix j, see which i(center i) is the index of min.
    return img_with_cluster_center 

# return new_center = the mean of all points in clusteri.
def find_new_center(k, point, img_with_cluster_center): #(center_with_point):
    new_center = []
    for i in np.arange(k):
        # point: kx(# of pixel)x3
        # point[img_with_cluster_center==i]: (# of pixel)x3
        new_center.append(point[img_with_cluster_center==i].mean(axis=0)) #3d, axis=1 fix j(3: RGB), compute mean
    return new_center

def kmean_step23(k, point, center):
    while 1:
        img_with_cluster_center = select_point_to_center(k, point, center)
        new_center = find_new_center(k, point, img_with_cluster_center)

        find_new_center_again = 0
        for j in np.arange(k):
            if not (np.array_equal(center, new_center)): # center[j] != new_center[j]: # check
                find_new_center_again = 1
            center = new_center
        if find_new_center_again == 0:
            break # have already found the center we want.
    return center, img_with_cluster_center # now, center is the new_center.

def kmean_function(k, image):
    print("k", k)
    # random50
    kmean_time = 0
    min_obj_function = sys.maxsize # set as maxint
    min_obj_func_center = []
    min_obj_func_img_with_cluster_center = []
    while kmean_time<2:
        kmean_time = kmean_time + 1
        # randomly initialize the cluster centers
        img_h, img_w = image.shape[:2] # img_image[img_h][img_w]
        img_h_random = np.random.randint(0, img_h, size=k) # the random index of img_h
        img_w_random = np.random.randint(0, img_w, size=k) # the random index of img_w
        center = []
        
        for i in np.arange(k):
            center.append(image[img_h_random[i]][img_w_random[i]])
   
        point = image.reshape(-1, 3) # point: row x 3

        new_center, img_with_cluster_center = kmean_step23(k, point, center)

        # choose the best result based on the objective function for each k.
        obj_function = 0 # for computing the objective function
        for i in np.arange(k):
            # point[img_with_cluster_center==i]-new_center[i]: a matrix, element:(point p(in clusteri)-new_center[i])
            # tmp_for_obj_function contains cluster i=0, ..(k-1), element: the square of (point p(in clusteri)-new_center[i])
            tmp = np.square(point[img_with_cluster_center==i]-new_center[i])
            tmp = np.sum(tmp)
            obj_function = obj_function + tmp
        # square: every element, sum all elements to a real number
        print("min_obj_function, obj_function", min_obj_function, obj_function)
        if min_obj_function > obj_function: 
            # if True, update min_obj_function, new_center, img_with_cluster_center
            min_obj_function = obj_function
            min_obj_func_center = new_center
            min_obj_func_img_with_cluster_center = img_with_cluster_center
        
    # show image
    # index i -> min_obj_func_img_with_cluster_center[i](which cluster j of point i) -> min_obj_func_center[j](the color)
    kmean_img = np.array(min_obj_func_center)[min_obj_func_img_with_cluster_center] 
    kmean_img = np.array(kmean_img.reshape(image.shape), dtype=np.uint8)

    return kmean_img

"""
kmean_5_image = kmean_function(5, img_image)
cv2.imwrite('output/2a_k_5.jpg', kmean_5_image)
"""
kmean_7_image = kmean_function(7, img_image)
cv2.imwrite('output/2a_k_7.jpg', kmean_7_image)

kmean_9_image = kmean_function(9, img_image)
cv2.imwrite('output/2a_k_9.jpg', kmean_9_image)

#cv2.imshow('2a_kmean_5_image', kmean_5_image)
#cv2.imshow('2a_kmean_7_image', kmean_7_image)
cv2.imshow('2a_kmean_9_image', kmean_9_image)
cv2.waitKey(0)
#"""

# k-means++
# randomly choose the first center
h, w, channel = img_image.shape

def kmean_plus_select_center(k, point): # point: (# of pixel)x3
    kmean_plus_center = []
    kmean_plus_center.append(point[np.random.randint(low=0, high=h*w-1, size=1)])
    center_num = 1
    while center_num<=k:
        print("center_num", center_num)
        center_num = center_num + 1
        # pick the next new center from dataset with prob
        D = 0 # initial
        dist = [] # dist[i]: dist[i]: the distance between point[i] and (all exist center). 
        for i in np.arange(len(point)):
            for j in np.arange(len(kmean_plus_center)):
                D = D + np.linalg.norm(point[i]-kmean_plus_center[j])
            dist.append(D)
        # randomly choose next center
        #cum
        dist = dist / np.sum(dist)
        dist = np.cumsum(dist)
        dist = np.insert(dist, 0, 0) 
        # dist = [0, ..., 1]
        ran_num = np.random.uniform(0, 1) # randomly choose from [0, 1]
        for i in np.arange(len(dist)-1):
            if dist[i] <= ran_num <= dist[i+1]: # find the interval
                kmean_plus_center.append(point[i])
                break 
    return kmean_plus_center

def kmean_plus_function(k, image):
    point = image.reshape(-1, 3)
    kmean_plus_center = kmean_plus_select_center(k, point)
    new_center, img_with_cluster_center = kmean_step23(k, point, kmean_plus_center)
    # show image
    # index i -> min_obj_func_img_with_cluster_center[i](which cluster j of point i) -> min_obj_func_center[j](the color)
    kmean_plus_img = np.array(new_center)[img_with_cluster_center] 
    kmean_plus_img = np.array(kmean_plus_img.reshape(image.shape), dtype=np.uint8)
    return kmean_plus_img

"""
kmean_plus_5_image = kmean_plus_function(5, img_image)
cv2.imwrite('output/2b_k_5.jpg', kmean_plus_5_image)

kmean_plus_7_image = kmean_plus_function(7, img_image)
cv2.imwrite('output/2b_k_7.jpg', kmean_plus_7_image)

kmean_plus_9_image = kmean_plus_function(9, img_image)
cv2.imwrite('output/2b_k_9.jpg', kmean_plus_9_image)

cv2.imshow('2b_kmean_plus_5_image', kmean_plus_5_image)
cv2.imshow('2b_kmean_plus_7_image', kmean_plus_7_image)
cv2.imshow('2b_kmean_plus_9_image', kmean_plus_9_image)
cv2.waitKey(0)
"""

# mean shift

def find_next_center_meanshift_RGB(re_image, center_now, bandwidth_square):
    row_re_image, col_re_image = re_image.shape
    diff_re_image_center_now = re_image - center_now
    dist_matrix = np.square(diff_re_image_center_now) # dist_matrix: (# of pixel) x 3(RGB)
    dist_matrix = np.sum(dist_matrix, axis=1) # dist_matrix: (# of pixel) 
    # now, element of dist_matrix is (R-Ri)^2+(G-Gi)^2+(B-Bi)^2

    weight_index = np.where(dist_matrix<=bandwidth_square) #, True, False) # 1:distance between image[i][j] and center is small enough, 0:o.w.

    re_weight = weight_index
    dist_multi_weight = re_image[re_weight]
    #dist_multi_weight = np.where(dist_multi_weight>0) # array(size:rowx3) with nonzero element in dist_multi_weight

    next_center = np.mean(dist_multi_weight, axis=0)
    next_center = next_center.astype(int)

    return next_center

img_h, img_w = img_image.shape[:2] # img_image[img_h][img_w]
img_h_resize, img_w_resize = int(img_h/16), int(img_w/16)

img_resize = cv2.resize(img_image, (img_w_resize, img_h_resize))
#img_resize = img_image
#cv2.imshow('2c', img_resize)
#cv2.waitKey(0)
re_image = img_resize.reshape(-1, 3)
re_image = np.array(re_image, dtype=np.int32)


def meanshift(re_image, bandwidth_square):
    img_meanshift = img_resize.copy() # initial
    
    for i in np.arange(img_h_resize): #*img_w_resize):
        for j in np.arange(img_w_resize):
            center = re_image[i*img_w_resize + j]  #[i][j]
            print("i, j, center", i, j, center)
            next_center = center
            
            while 1 :
                next_center = find_next_center_meanshift_RGB(re_image, center, bandwidth_square)
                print("next_center", next_center)
                if not np.array_equal(center, next_center):
                    center = next_center
                else: 
                    break
            
            add_next_center = next_center[:3]
            img_meanshift[i][j] = add_next_center
            #print(img_meanshift, i) #, j)

    img_meanshift = np.array(img_meanshift, dtype=np.uint8)
    #img_meanshift = np.array(img_meanshift.reshape(re_image.shape), dtype=np.uint8)
    return img_meanshift


#img_RGB_featurespace = 
#"""
def plotrgb(ori_image, new_image, img_h_resize, img_w_size):
    fig = plt.figure()
    axis = plt.axes(projection="3d")
    x = []
    y = []
    z = []
    color = []
    for i in np.arange(img_h_resize):
        for j in np.arange(img_w_resize):
            ori_pixel = ori_image[i][j]
            new_pixel = new_image[i][j] #[i, j]
            newcolor = (new_pixel[0]/255, new_pixel[1]/255, new_pixel[2]/255)
            #if not newcolor in color :
            x.append(ori_pixel[0])
            y.append(ori_pixel[1])
            z.append(ori_pixel[2])
            color.append(newcolor)
    axis.scatter(x, y, z, c = color, marker=".")       
    #axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=img_meanshift_RGB.tolist(), marker=".")
    axis.set_xlabel("Red")
    axis.set_ylabel("Green")
    axis.set_zlabel("Blue")
    plt.show()
#"""
plotrgb(img_resize, img_resize, img_h_resize, img_w_resize)
img_meanshift_RGB = meanshift(re_image, 1000) #RGB
cv2.imwrite('output/meanshift_rgb1000.jpg', img_meanshift_RGB)
plotrgb(img_resize, img_meanshift_RGB, img_h_resize, img_w_resize)

img_meanshift_RGB = meanshift(re_image, 400) #RGB
cv2.imwrite('output/meanshift_rgb400.jpg', img_meanshift_RGB)
img_meanshift_RGB = meanshift(re_image, 8000) #RGB
cv2.imwrite('output/meanshift_rgb8000.jpg', img_meanshift_RGB)


# RGBxy
re_image_RGBxy = [] # = img_resize.copy()
#xx, yy = np.meshgrid(np.arange(img_w_resize), np.arange(img_h_resize))
for i in np.arange(img_h_resize):
    for j in np.arange(img_w_resize):
        re_image_RGBxy = np.append(re_image_RGBxy, [img_resize[i][j][0], img_resize[i][j][1], img_resize[i][j][2], i, j])
        


re_image_RGBxy = re_image_RGBxy.reshape(-1, 5)
re_image_RGBxy = np.array(re_image_RGBxy, dtype=np.int32)
img_meanshift_RGBxy = meanshift(re_image_RGBxy, 50)

cv2.imwrite('output/meanshift_rgbxy.jpg', img_meanshift_RGBxy)

