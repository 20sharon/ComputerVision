# hw2_1
# A = x1x'1 x1y'1 x1 y1x'1 y1y'1 y1 x'1 y'1 1
#     :                                     :
#     xmx'm xmy'm xm ymx'm ymy'm ym x'm y'm 1

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


file1 = open("assets\pt_2D_1.txt", 'r')
file2 = open("assets\pt_2D_2.txt", 'r')
img1 = cv2.imread('assets\image1.jpg')
img2 = cv2.imread('assets\image2.jpg')


point_1_list = []
point_2_list = []
row_num_1 = file1.readline()
row_num_2 = file2.readline()

def Sum_of_Euclidean_Distance(point, center_point):
    dist = 0
    for i in np.arange(int(row_num_1)):
        dist = dist + (point[i]-center_point)@(point[i]-center_point).T 
    return dist

# known point, find the fundamental matrix
def find_fundamental_matrix(img1_point_list, img2_point_list):
    A = []
    for i in np.arange(int(row_num_1)):
        x_1, y_1 = img1_point_list[i][0], img1_point_list[i][1]
        x_2, y_2 = img2_point_list[i][0], img2_point_list[i][1]
        add_row = np.array([x_1*x_2, x_1*y_2, x_1, y_1*x_2, y_1*y_2, y_1, x_2, y_2, 1])
        if i==0:
            A = np.hstack((A, add_row))
        else:
            A = np.vstack((A, add_row))

    # find SVD of AT A
    # A:MxN, full_matrices=1 means U:MxM and V:NxN. Otherwise, U:MxK, V:KxN, K=min(M,N)
    #  compute_uv=1 means compute U, sigma, VT. Otherwise, only compute sigma.
    U, sigma, VT = np.linalg.svd(A, full_matrices=1, compute_uv=1)
    # Entries of F are the elements of column of V corresponding to the least singular value
    # A = UxSxVT, where V = (v1, v2, ...vn), the column is vn, VT is the transport of V
    F = (VT[8]).reshape(3, 3)
    # Enforce rank2 constraint
    F_U, F_S, F_VT = np.linalg.svd(F)
    F = F_U@np.diag([F_S[0], F_S[1], 0])@F_VT
    return F

# epipolar line
def plot_epipolar_line(img, w, F, point_list, plot_point_list):
    colormap = plt.get_cmap("rainbow")
    for i in np.arange(int(row_num_1)):
        color = colormap(i/len(point_list))
        # point1(0, -c/b),point2(w, -(a*w+c)/b) are on line:ax+by+c=0 
        a, b, c = F@(np.array([point_list[i][0], point_list[i][1], 1]).T)
        point1 = (0, -c/b)
        point2 = (w, -(a*w+c)/b)
        plt.plot(*zip(point1, point2), color=color)
        plt.plot(plot_point_list[i][0], plot_point_list[i][1], ".", color=color)
    plt.imshow(img)
    plt.show()
    return

# shortest distance between a point:(p1, p2) and a line:ax+by+c=0
# point are point_on_img, line created by F@point_create_line
def total_dist(F, point_create_line, num_of_point, point_on_img):
    total_dist_in_accu = 0
    for i in np.arange(num_of_point):
        a, b, c = F@(np.array([point_create_line[i][0], point_create_line[i][1], 1]).T)
        p1, p2 = point_on_img[i][0], point_on_img[i][1]
        d = abs((a*p1 + b*p2 + c)) / (np.sqrt(a*a + b*b))
        total_dist_in_accu = total_dist_in_accu + d
    return total_dist_in_accu


# read points from the files
for i in np.arange(int(row_num_1)):
    line1 = file1.readline() # type str
    line2 = file2.readline()

    point1 = line1.split() #np.char.split(line1) # point1 is a list
    point2 = line2.split() #np.char.split(line2) # point2 is a list
    
    x_1, y_1 = float(point1[0]), float(point1[1])
    x_2, y_2 = float(point2[0]), float(point2[1])

    if i == 0:
        point_1_list = np.hstack((point_1_list, np.array([x_1, y_1])))
        point_2_list = np.hstack((point_2_list, np.array([x_2, y_2])))
    else:
        point_1_list = np.vstack((point_1_list, np.array([x_1, y_1])))
        point_2_list = np.vstack((point_2_list, np.array([x_2, y_2])))

# normalized
x, y, channel = img1.shape
image1_center_x, image1_center_y = x/2, y/2

# mean squared distance between center and the data points is 2 pixels
dist1 = Sum_of_Euclidean_Distance(point_1_list, [image1_center_x, image1_center_y])
dist2 = Sum_of_Euclidean_Distance(point_2_list, [image1_center_x, image1_center_y])

s1 = np.sqrt(2/dist1)
s2 = np.sqrt(2/dist2)
T1 = np.diag([s1, s1, 1])@[[1, 0, -image1_center_x], 
                           [0, 1, -image1_center_y],
                           [0, 0, 1]]
T2 = np.diag([s2, s2, 1])@[[1, 0, -image1_center_x], 
                           [0, 1, -image1_center_y], 
                           [0, 0, 1]]

normalized_point_1_list = []
normalized_point_2_list = []

for i in np.arange(int(row_num_1)):
    add_normalized_point_1 = T1@[point_1_list[i][0], point_1_list[i][1], 1]
    add_normalized_point_2 = T2@[point_2_list[i][0], point_2_list[i][1], 1]
    if i == 0:
        normalized_point_1_list = np.hstack((normalized_point_1_list, add_normalized_point_1)) #test array
        normalized_point_2_list = np.hstack((normalized_point_2_list, add_normalized_point_2))
    else:
        normalized_point_1_list = np.vstack((normalized_point_1_list, add_normalized_point_1)) #test array
        normalized_point_2_list = np.vstack((normalized_point_2_list, add_normalized_point_2))

non_normalized_F = find_fundamental_matrix(point_1_list, point_2_list)
normalized_F = find_fundamental_matrix(normalized_point_1_list, normalized_point_2_list)


print("non normal F")
print(non_normalized_F)
normalized_F = T1.T@normalized_F@T2
print("normalized F")
print(normalized_F)


# plot epipolar line
# on img2, epipolar line is F.T@P
plot_epipolar_line(img2, y, non_normalized_F.T, point_1_list, point_2_list)
plot_epipolar_line(img1, y, non_normalized_F, point_2_list, point_1_list)
plot_epipolar_line(img2, y, normalized_F.T, point_1_list, point_2_list)
plot_epipolar_line(img1, y, normalized_F, point_2_list, point_1_list) 

# the accuracy
avg_dist_between_non_normalized_f_and_point = (total_dist(non_normalized_F.T, point_1_list, int(row_num_1), point_2_list)+total_dist(non_normalized_F, point_2_list, int(row_num_1), point_1_list))/(2*int(row_num_1))
avg_dist_between_normalized_f_and_point = (total_dist(normalized_F.T, point_1_list, int(row_num_1), point_2_list)+total_dist(normalized_F, point_2_list, int(row_num_1), point_1_list))/(2*int(row_num_1))
print("accuracy of non_normalized F", avg_dist_between_non_normalized_f_and_point)
print("accuracy of normal F", avg_dist_between_normalized_f_and_point)

file1.close()
file2.close()