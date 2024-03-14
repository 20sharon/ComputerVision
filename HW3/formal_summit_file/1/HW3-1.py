import cv2
import numpy as np
import matplotlib.pyplot as plt

img_book1 = cv2.imread('1-book1.jpg')
img_book2 = cv2.imread('1-book2.jpg')
img_book3 = cv2.imread('1-book3.jpg')
img_image = cv2.imread('1-image.jpg')


def SIFT_object_recognition(img_1, img_2): # img1->image, img2->book1, book2, book3
    sift = cv2.SIFT_create() # construct a SIFT object
    kps_1, des_1 = sift.detectAndCompute(img_1, None) # Given a set of Keypoint,compute descriptors.
    kps_2, des_2 = sift.detectAndCompute(img_2, None)

    kps_1_len = len(kps_1)
    kps_2_len = len(kps_2)


    # Brute-force matching and ratio test
    def Brute_force_matching(): #keypoint_num, descriptor_num, descriptor_num_idx):
        distance_num = ()
        
        for i in range(kps_1_len): # image
            # init tmp_set
            tmp_set = []

            for j in range(kps_2_len): # book
                # consider the min distance between des_1[i] and des_2[j]
                dist_compute = np.linalg.norm(des_1[i]-des_2[j]) #descriptor_num[j]) # init
                # tmp_set for collecting all (idx_j, distance_between_i_and_j) 
                tmp_set.append((cv2.DMatch(i, j, dist_compute))) 
            # for i, append the nearest one B and the second nearest one C from A   
            # select in tmp_set
            tmp_set = sorted(tmp_set, key = lambda x:x.distance)
            tmp_set = tuple(tmp_set)
            distance_num = ((tmp_set[0], tmp_set[1]), ) + distance_num

        return distance_num #keypoint_num, descriptor_num, descriptor_num_idx, distance_num


    """keypoint_book, descriptor_book, descriptor_book_idx, """
    distance_book = Brute_force_matching()#kps_2, des_2,)
    print("len", len(distance_book))

    # apply ratio test
    def ratio_test(distance_num):
        matches_num = []
        for m, n in distance_num:
            if m.distance < 0.85*n.distance: # 0.5
                # threshold = 600
                if m.distance <= 600:
                    matches_num.append([m]) 
        return matches_num

    matches = ratio_test(distance_book)

    

    print("len", len(matches))


    # sort in the order of the distance
    matches = sorted(matches, key = lambda x:x[0].distance) #x[0].distance)

    matchesall = matches[:510] #[:510]

    img_match = cv2.drawMatchesKnn(img_1, kps_1, img_2, kps_2, matchesall, outImg=None, matchColor = (255, 0, 0))
    plt.imshow(img_match), plt.show()

    return kps_1, kps_2, matchesall, img_match


#SIFT_object_recognition(test1, test2)
image1_kps, book1_kps, book1_image_match, img_book1_1a_ans = SIFT_object_recognition(img_image, img_book1)
image2_kps, book2_kps, book2_image_match, img_book2_1a_ans = SIFT_object_recognition(img_image, img_book2)
image3_kps, book3_kps, book3_image_match, img_book3_1a_ans = SIFT_object_recognition(img_image, img_book3)

cv2.imwrite('output/1a_ans_img_book1.jpg', img_book1_1a_ans)
cv2.imwrite('output/1a_ans_img_book2.jpg', img_book2_1a_ans)
cv2.imwrite('output/1a_ans_img_book3.jpg', img_book3_1a_ans)



# RANSAC homography , parameter:loop_time, inlier_threshold
# homography
# A = [[x_1 y_1 1 0   0   0 -x_2*x_1 -x_2*y_1 -x_2]
#      [0   0   0 x_1 y_1 1 -y_2*x_1 -y_2*y_1 -y_2]]
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

def dist(img_point, H_matrix, book_point): # the distance between the estimate point(H@book_point) and real point(image_point)
        esti_point_in_img = H_matrix@(np.array([book_point[0], book_point[1], 1]).T)
        esti_point_in_img[0] = esti_point_in_img[0] / esti_point_in_img[2]
        esti_point_in_img[1] = esti_point_in_img[1] / esti_point_in_img[2]
        esti_point_in_img[2] = 1
        dist = np.linalg.norm([img_point[0], img_point[1], 1] - esti_point_in_img.T)
        return dist

len_book1_image_match = len(book1_image_match)
len_book2_image_match = len(book2_image_match)
len_book3_image_match = len(book3_image_match)


# initial
corner_point_in_book1 = [[96, 220],[999, 224], [987, 1333], [119, 1349]]
corner_point_in_book2 = [[65, 122], [1045, 110], [1043, 1344], [74, 1352]]
corner_point_in_book3 = [[124, 182], [996, 177], [983, 1392], [129, 1397]]


def point_H_line(corner_point_in_book, H, img_image, img_book): # point -> H -> point(image)
    img_with_corner = img_image.copy()
    img_book_with_corner = img_book.copy()
    corner_list_in_img = [] #leftup, rightup, rightdown, leftdown
    corner_list_in_book = []
    for i in np.arange(4):
        corner_point_in_img = H@(np.array([corner_point_in_book[i][0], corner_point_in_book[i][1], 1]).T)
        corner_point_in_img[0] = corner_point_in_img[0] / corner_point_in_img[2]
        corner_point_in_img[1] = corner_point_in_img[1] / corner_point_in_img[2]
        corner_list_in_img.append((int(corner_point_in_img[0]), int(corner_point_in_img[1])))
        corner_list_in_book.append((int(corner_point_in_book[i][0]), int(corner_point_in_book[i][1])))
    img_with_corner = cv2.line(img_with_corner, corner_list_in_img[0], corner_list_in_img[1], color=(255, 0, 0), thickness=5)
    img_with_corner = cv2.line(img_with_corner, corner_list_in_img[1], corner_list_in_img[2], color=(255, 0, 0), thickness=5)
    img_with_corner = cv2.line(img_with_corner, corner_list_in_img[2], corner_list_in_img[3], color=(255, 0, 0), thickness=5)
    img_with_corner = cv2.line(img_with_corner, corner_list_in_img[3], corner_list_in_img[0], color=(255, 0, 0), thickness=5)

    img_book_with_corner = cv2.line(img_book_with_corner, corner_list_in_book[0], corner_list_in_book[1], color=(255, 0, 0), thickness=5)
    img_book_with_corner = cv2.line(img_book_with_corner, corner_list_in_book[1], corner_list_in_book[2], color=(255, 0, 0), thickness=5)
    img_book_with_corner = cv2.line(img_book_with_corner, corner_list_in_book[2], corner_list_in_book[3], color=(255, 0, 0), thickness=5)
    img_book_with_corner = cv2.line(img_book_with_corner, corner_list_in_book[3], corner_list_in_book[0], color=(255, 0, 0), thickness=5)
    return img_with_corner, img_book_with_corner

def ransac_homography(len_book_image_match, book_image_match, image_kps, book_kps, img_book, inlier_threshold, corner_point_in_book):
    loop_time = 0
    best_H = []
    best_inlier_of_H = []
    best_inlier_len = 0 # the length of best_inlier_of_H
    while loop_time<=2000: # 2000 # find the best homography matrix among loop_time.
        loop_time = loop_time + 1
        # randomly choose 4 points
        ran_num = np.random.randint(low=0, high=len_book_image_match, size=4)
        #print(ran_num)
        image_4_point = [] # the four point we randomly choose in img image
        book_4_point = [] # the four point we randomly choose in img book1
        for i in np.arange(4):
            queryidx = book_image_match[ran_num[i]][0].queryIdx # img1 -> image
            trainidx = book_image_match[ran_num[i]][0].trainIdx # img2 -> book
            image_kps_x, image_kps_y = image_kps[queryidx].pt
            book_kps_x, book_kps_y = book_kps[trainidx].pt
            #if i == 0:
            image_4_point.append([image_kps_x, image_kps_y])
            book_4_point.append([book_kps_x, book_kps_y])

        # find homography matrix H
        H = Find_Homography(image_4_point, book_4_point) # book - H > image

        # count inlier
        #inlier_threshold = 50
        inlier_match = []
        for i in np.arange(len_book_image_match): # check all match
            queryidx = book_image_match[i][0].queryIdx # queryidx is the index of image's kps
            trainidx = book_image_match[i][0].trainIdx # trainidx is the index of book's kps
            image_kps_x, image_kps_y = image_kps[queryidx].pt # the position of image's kps
            book_kps_x, book_kps_y = book_kps[trainidx].pt # the position of book's kps
            
            # compute the distance of match points 
            distance = dist([image_kps_x, image_kps_y], H, [book_kps_x, book_kps_y])
            print("distance", distance)
            if distance < inlier_threshold : # True->is inlier
                inlier_match.append(book_image_match[i])
        
        # check whether or not to update best_H and best_inlier_of_H
        if(best_inlier_len < len(inlier_match)): 
            # inlier become more, need to update
            best_H = H
            best_inlier_of_H = inlier_match
            best_inlier_len = len(best_inlier_of_H)
    print("the best matching homography transformation")
    print(best_H)

    # draw line
    img_image_with_line, img_book_with_line = point_H_line(corner_point_in_book, best_H, img_image, img_book)

    img_match = cv2.drawMatchesKnn(img_image_with_line, image_kps, img_book_with_line, book_kps, best_inlier_of_H, outImg=None, matchColor = (255, 0, 0))

    plt.imshow(img_match), plt.show()
    return best_inlier_of_H, best_inlier_len, best_H, img_match



best_inlier_of_H1, best_inlier_len_book1, best_H1, img_match_book1_ransac = ransac_homography(len_book1_image_match, book1_image_match, image1_kps, book1_kps, img_book1, 50, corner_point_in_book1)
best_inlier_of_H2, best_inlier_len_book2, best_H2, img_match_book2_ransac = ransac_homography(len_book2_image_match, book2_image_match, image2_kps, book2_kps, img_book2, 20, corner_point_in_book2)
best_inlier_of_H3, best_inlier_len_book3, best_H3, img_match_book3_ransac = ransac_homography(len_book3_image_match, book3_image_match, image3_kps, book3_kps, img_book3, 30, corner_point_in_book3)


cv2.imwrite('output/1b_ans_img_book1.jpg', img_match_book1_ransac)
cv2.imwrite('output/1b_ans_img_book2.jpg', img_match_book2_ransac)
cv2.imwrite('output/1b_ans_img_book3.jpg', img_match_book3_ransac)

img_image_b = img_image.copy()

def draw_deviation_vector(img_image_b, best_inlier_of_H, best_inlier_len, best_H, book_kps, image_kps):
    # show the deviation vector
    for i in np.arange(best_inlier_len):
        # find the position of P in book
        trainidx = best_inlier_of_H[i][0].trainIdx # trainidx is the index of book's kps
        book_kps_x, book_kps_y = book_kps[trainidx].pt # the position of book's kps
        # find the position in image, P's corresponding feature point
        queryidx = best_inlier_of_H[i][0].queryIdx # queryidx is the index of image's kps
        image_kps_x, image_kps_y = image_kps[queryidx].pt # the position of image's kps
        # find the position in image, HP (transformed feature point)
        transformed_point_in_img = best_H@(np.array([book_kps_x, book_kps_y, 1]).T)
        transformed_point_in_img[0] = transformed_point_in_img[0] / transformed_point_in_img[2]
        transformed_point_in_img[1] = transformed_point_in_img[1] / transformed_point_in_img[2]
        # draw arrowed line
        img_image_b = cv2.arrowedLine(img_image_b, (int(image_kps_x), int(image_kps_y)), (int(transformed_point_in_img[0]), int(transformed_point_in_img[1])), color=(0, 0, 255), thickness=2)
    return img_image_b

img_image_b = draw_deviation_vector(img_image_b, best_inlier_of_H1, best_inlier_len_book1, best_H1, book1_kps, image1_kps)
img_image_b = draw_deviation_vector(img_image_b, best_inlier_of_H2, best_inlier_len_book2, best_H2, book2_kps, image2_kps)
img_image_b = draw_deviation_vector(img_image_b, best_inlier_of_H3, best_inlier_len_book3, best_H3, book3_kps, image3_kps)
cv2.imwrite('output/1b_ans_image.jpg', img_image_b)