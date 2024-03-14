import cv2
import numpy as np
import copy
import os
import matplotlib.pyplot as plt
#import math
# mouse callback function
def mouse_callback(event, x, y, flags, param):
    
    global corner_list
    if event == cv2.EVENT_LBUTTONDOWN:  
        if(len(corner_list)<4):
            corner_list.append((x,y))
        
def Find_Homography(world,camera):
    '''
    given corresponding point and return the homagraphic matrix 
    '''
    # world(left) x_2 y_2 <- H -- camera(right) x_1 y_1
    # x_1 y_1 1 0   0   0 -x_2*x_1 -x_2*y_1 -x_2
    # 0   0   0 x_1 y_1 1 -y_2*x_1 -y_2*y_1 -y_2
    A = []
    for i in np.arange(4):
        x_1, y_1 = camera[i][0], camera[i][1]
        x_2, y_2 = world[i][0], world[i][1]
        add_row = [[x_1, y_1, 1, 0, 0, 0, -x_2*x_1, -x_2*y_1, -x_2],
                   [0, 0, 0, x_1, y_1, 1, -y_2*x_1, -y_2*y_1, -y_2]]
        if i==0:
            A = add_row
        else:
            A = np.vstack((A, add_row))
    A_U, A_S, A_VT = np.linalg.svd(A)
    H = (A_VT[8]).reshape(3, 3)
    return H

if __name__=="__main__":
    
    img_src = cv2.imread("assets/post.png") 
    src_H,src_W,_=img_src.shape
    # print(H,W)
    file_path="./1/output"
    img_tar = cv2.imread("assets/display.jpg") 
    
    cv2.namedWindow("Interative window")
    cv2.setMouseCallback("Interative window", mouse_callback)
    cv2.setMouseCallback("Interative window", mouse_callback)
    
    corner_list=[]
    while True:
        fig=img_tar.copy()
        key = cv2.waitKey(1) & 0xFF
        
        
        if(len(corner_list)==4):
            # implement the inverse homography mapping and bi-linear interpolation 
            world = corner_list
            print("corner_list")
            print(corner_list)
            camera = [[0, 0],
                      [src_W, 0], 
                      [src_W, src_H], 
                      [0, src_H]]
            H_map_CVimage_to_screen = Find_Homography(world, camera) # world <-H- camera
            inverse_H = Find_Homography(camera, world) # world -inverse_H -> camera
            print("H(map CV image to screen)")
            print(H_map_CVimage_to_screen)
            print("inverse H")
            print(inverse_H)

            array_corner = np.array(corner_list)
            min_x = np.min(array_corner[:, 0])
            max_x = np.max(array_corner[:, 0])
            min_y = np.min(array_corner[:, 1])
            max_y = np.max(array_corner[:, 1])

            for i in np.arange(min_x, max_x+1): # [min_x, max_x]
                for j in np.arange(min_y, max_y+1): # [min_y, max_y]
                    point_in_camera = inverse_H@(np.array([i, j, 1]).T)
                    point_in_camera[0] = point_in_camera[0]/point_in_camera[2]
                    point_in_camera[1] = point_in_camera[1]/point_in_camera[2]
        
                    if (0<=point_in_camera[0] and point_in_camera[0]<src_W-1 and
                        0<=point_in_camera[1] and point_in_camera[1]<src_H-1):
                        # point is in img of camera
                        # find color in (point_in_camera[0], point_in_camera[1]) by bi-linear interpolation
                        # tmp1:(int(point_in_camera[0]), point_in_camera[1]) tmp2:(int(point_in_camera[0])+1, point_in_camera[1])
                        # imgsrc00(img[int(point[1])][int(point[0])])     tmp1   imgsrc01(img[int(point[1])+1][int(point[0])])
                        #                                                  p
                        # imgsrc10(img[int(point[1])][int(point[0])+1])   tmp2   imgsrc11(img[int(point[1])+1][int(point[0])+1])
                        # img[j][i] = color
                        imgsrc00 = img_src[int(point_in_camera[1])][int(point_in_camera[0])]
                        imgsrc01 = img_src[int(point_in_camera[1])+1][int(point_in_camera[0])]
                        imgsrc10 = img_src[int(point_in_camera[1])][int(point_in_camera[0])+1]
                        imgsrc11 = img_src[int(point_in_camera[1])+1][int(point_in_camera[0])+1]
                        tmp1 = imgsrc00*(int(point_in_camera[1])+1 - point_in_camera[1]) + imgsrc01*(point_in_camera[1] - int(point_in_camera[1]))
                        tmp2 = imgsrc10*(int(point_in_camera[1])+1 - point_in_camera[1]) + imgsrc11*(point_in_camera[1] - int(point_in_camera[1]))
                        color_value = tmp1*(int(point_in_camera[0])+1 - point_in_camera[0]) + tmp2*(point_in_camera[0] - int(point_in_camera[0]))
                        fig[j][i] = color_value 
            # compute vanishing point(v1, v2)
            # the intersection of line((corner_list[0]), (corner_list[1])) and 
            # line((corner_list[2]), (corner_list[3]))
            L1 = np.cross((corner_list[0][0], corner_list[0][1], 1), (corner_list[1][0], corner_list[1][1], 1))
            L2 = np.cross((corner_list[2][0], corner_list[2][1], 1), (corner_list[3][0], corner_list[3][1], 1))
            kv1, kv2, k = np.cross(L1, L2)
            v1 = kv1/k
            v2 = kv2/k
            print("vanishing point's coordinate (", int(v1), int(v2), ")")
            # print vanishing point
            cv2.circle(fig, center=(int(v1), int(v2)), radius=5, color=(0, 255, 0), thickness=-1)
            # print corner
            cv2.circle(fig, corner_list[0], 5, color=(0, 255, 0), thickness=-1)
            cv2.circle(fig, corner_list[1], 5, color=(0, 255, 0), thickness=-1)
            cv2.circle(fig, corner_list[2], 5, color=(0, 255, 0), thickness=-1)
            cv2.circle(fig, corner_list[3], 5, color=(0, 255, 0), thickness=-1)
            # print line
            cv2.line(fig, corner_list[0], corner_list[1], color=(75, 0, 130), thickness=3)
            cv2.line(fig, corner_list[1], corner_list[2], color=(75, 0, 130), thickness=3)
            cv2.line(fig, corner_list[2], corner_list[3], color=(75, 0, 130), thickness=3)
            cv2.line(fig, corner_list[3], corner_list[0], color=(75, 0, 130), thickness=3)
            # put the image fig in homography.png
            cv2.imwrite('output/homography.png', fig)
            #cv2.imshow('result', fig)
            #cv2.waitKey(50)
            # pass
            
        # quit 
        if key == ord("q"):
            break
    
        # reset the corner_list
        if key == ord("r"):
            corner_list=[]
        # show the corner list
        if key == ord("p"):
            print(corner_list)
        cv2.imshow("Interative window", fig)

    cv2.imwrite(os.path.join(file_path,"homography.png"),fig)
    cv2.destroyAllWindows()