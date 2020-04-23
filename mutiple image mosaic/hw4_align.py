import cv2
import numpy as np
import os
import math
import copy
import sys

'''for this project opencv version is 4.1.0.25
    if using 3.4.2.16, you will get different
    results for some reason, I think it is because
    they update the detector so the keypoints will
    different and the percent of survive inliers
    will different
'''

'''read image from folder'''
def read_image(foldername):
    image_list=os.listdir('./'+foldername)
    image_list.sort()
    im_list=[]
    for i in range(len(image_list)):
        im=cv2.imread('./'+foldername+'/'+image_list[i],-1)
        im_list.append(im)
    return im_list,image_list

'''extract keypoints and match them'''
def feature_matching(img1,img2):
    orb=cv2.ORB_create()
    kp1,des1 = orb.detectAndCompute(img1, None)
    kp2,des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    return kp1,kp2,des1,des2,matches

'''using matched keypoints to calculate F and H,and using RANSC to eliminate
    some wrong match
   and if I get more than 70% inliers then i think most points can match
   if 20%-70% inliers survive, i think small parts of points can match and it
   can be accepted
   But if below 20% inliers, then Most of points cannot match and this is a
   terrible match
'''
def find_homography(img1,img2,kp1,kp2,matches,im_list1,im_list2,out_dir):
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches[:] ])
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches[:] ])
    F, F_mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC)
    matchesMask = F_mask.ravel().tolist()
    draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = None,
                   matchesMask = matchesMask, 
                   flags = 2)
    
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None,**draw_params)
    filename=(im_list1.split('.')[0]+'_'+
                  im_list2.split('.')[0]+'_'+'Fmatching'+
                '.'+im_list1.split('.')[1])
    cv2.imwrite('./'+out_dir+'/'+filename,img3)
    H, H_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = H_mask.ravel().tolist()
    draw_params = dict(matchColor = (0,255,0), 
                   singlePointColor = None,
                   matchesMask = matchesMask, 
                   flags = 2)
    img4 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None,**draw_params)
    filename=(im_list1.split('.')[0]+'_'+
                  im_list2.split('.')[0]+'_'+'Hmatching'+
                '.'+im_list1.split('.')[1])
    cv2.imwrite('./'+out_dir+'/'+filename,img4)
    if np.sum(F_mask)>0.7*len(matches):
        print('More than 70% percent inliers survive in Fundamental Matrix')
        print('We calculate the Homography matrix')
        if np.sum(H_mask)>0.7*len(matches):
            print('Still More than 70% percent inliers survive')
            print('Most parts of points can match')
            H=H
        elif np.sum(H_mask)>0.2*len(matches) and np.sum(H_mask)<0.7*len(matches):
            print('About 20%-70% percent keypoints survive')
            print('Small parts of points can match')
            H=H
        else:
            print('Below 20% percent keypoints survive')
            print('Most keypoints cannot match')
            H=np.zeros((3,3))
    elif np.sum(F_mask)>0.2*len(matches) and np.sum(F_mask)<0.7*len(matches):
        print('20%-70% percent inliers survive in Fundamental Matrix')
        print('We calculate the Homography matrix')
        if np.sum(H_mask)>0.2*len(matches) and np.sum(H_mask)<0.7*len(matches):
            print('About 20%-70% percent keypoints survive')
            print('Small parts of points can match')
            H=H
        else:
            print('Below 20% percent keypoints survive')
            print('Most keypoints cannot match')
            H=np.zeros((3,3))
    else:
            print('Below 20% percent keypoints survive in Fundamental Matrix')
            print('Most keypoints cannot match')
            H=np.zeros((3,3))
    return H

'''this is just same with find_homography but it will not print anything'''
def find_homography_for_multi_image(kp1,kp2,matches):
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches[:] ])
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches[:] ])
    H, H_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    return H

'''using cv2.warpperspective to map image1 onto image2 and return the origin'''
def warp_image(image,Homography):
    h,w,z = image.shape
    p = np.array([[0, w, w, 0], [0, 0, h, h], [1, 1, 1, 1]])
    p_prime = np.dot(Homography, p)
    yrow = p_prime[1] / p_prime[2]
    xrow = p_prime[0] / p_prime[2]
    ymin = min(yrow)
    xmin = min(xrow)
    ymax = max(yrow)
    xmax = max(xrow)
    new_mat = np.array([[1, 0, -1 * xmin], [0, 1, -1 * ymin], [0, 0, 1]])
    Homography = np.dot(new_mat, Homography)
    height = int(round(ymax - ymin))
    width = int(round(xmax - xmin))
    size = (width, height)
    warped = cv2.warpPerspective(image,Homography, size)
    origin=(int(xmin), int(ymin))
    return warped,origin

''' create mosaic
    i fix one image and fix its origin with (0,0),then i warp another image
    The origin of warped image will change, then i add transition to Homography
    matrix to get a full warped image and at the end i stitch two image with 
    average blending
'''
def stitich(images,origins):
    for i in range(0, len(origins)):
        if origins[i] == (0, 0):
            central_index = i
            break
    central_image = images[central_index]
    central_origin = origins[central_index]
    zipped = list(zip(origins, images))
    func = lambda x: math.sqrt(x[0][0] ** 2 + x[0][1] ** 2)
    dist_sorted = sorted(zipped, key=func, reverse=True)
    x_sorted = sorted(zipped, key=lambda x: x[0][0])
    y_sorted = sorted(zipped, key=lambda x: x[0][1])
    if x_sorted[0][0][0] > 0:
        cent_x = 0  
    else:
        cent_x = abs(x_sorted[0][0][0])
    if y_sorted[0][0][1] > 0:
        cent_y = 0  
    else:
        cent_y = abs(y_sorted[0][0][1])
    spots = []
    for origin in origins:
        spots.append((origin[0]+cent_x, origin[1] + cent_y))
    zipped = list(zip(spots, images))
    total_height = 0
    total_width = 0
    for spot, image in zipped:
        total_width = max(total_width, spot[0]+image.shape[1])
        total_height = max(total_height, spot[1]+image.shape[0])
    stitch = np.zeros((total_height, total_width,3), np.uint8)
    k=1
    for image in dist_sorted:
        offset_y = image[0][1] + cent_y
        offset_x = image[0][0] + cent_x
        end_y = offset_y + image[1].shape[0]
        end_x = offset_x + image[1].shape[1]
        if k==1:
            stitch[offset_y:end_y, offset_x:end_x,:] = image[1]
            k=k+1
        else:
            for c in range(3):
                for i in range(offset_y,end_y,1):
                    for j in range(offset_x,end_x,1):
                        if stitch[i,j,c]!=0 and image[1][i-offset_y,j-offset_x,c]!=0:
                            alpha=0.5
                            stitch[i,j,c]=np.uint8(((1-alpha)*stitch[i,j,c]+
                                  alpha*image[1][i-offset_y,j-offset_x,c]))
                        elif stitch[i,j,c]!=0 and image[1][i-offset_y,j-offset_x,c]==0:
                            stitch[i,j,c]=stitch[i,j,c]
                        else:
                            stitch[i,j,c]=image[1][i-offset_y,j-offset_x,c]
    return stitch

'''this pair_image function to determine if two pair can match each other
    if they match, then output mosaic pairs image
'''
def pair_image(i,j,im_list,image_list,min_keypoints,Node,out_dir):
    kp1,kp2,des1,des2,matches=feature_matching(im_list[i],im_list[j])
    if len(matches)>min_keypoints:
        H=find_homography(im_list[i],im_list[j],kp1,kp2,matches,image_list[i],image_list[j],out_dir)
        if H[2,2]==0:
            print('No warp')
            warped=0
        else:
            warped,origin=warp_image(im_list[i],H)
            pano=stitich([warped,im_list[j]],[origin,(0,0)])
            filename=(image_list[i].split('.')[0]+'_'+
                  image_list[j].split('.')[0]+
                '.'+image_list[i].split('.')[1])
            cv2.imwrite('./'+out_dir+'/'+filename,pano)
            Node.append([i,j])
    return Node,warped

'''this is same with pair_image but it will not print anything and judge anything'''
def pair_(i,j,im_list):
    kp1,kp2,des1,des2,matches=feature_matching(im_list[i],im_list[j])
    H=find_homography_for_multi_image(kp1,kp2,matches)
    warped,origin=warp_image(im_list[i],H)
    return warped,origin

'''create multi image mosaic
    Here, i consider all conditions.
    What i do here is, i let the node which has the most edges as anchor image
    Then i find all direct connected components and if they do not have any 
    other connected components then i map them onto anchor, if they have other
    connected components,i map other components onto it first and then i map
    them onto anchor image
'''
def multi_image_mosaic(Node,im_list,out_dir,image_list):
    Node_list=list(np.array(Node).reshape(np.array(Node).shape[0]*np.array(Node).shape[1]))
    number_of_node=np.arange(len(im_list))
    number_of_edge=np.zeros(len(im_list))
    for i in range(len(number_of_node)):
        number_of_edge[i]=np.sum(np.array(Node_list)==number_of_node[i])
    if np.sum(number_of_edge==max(number_of_edge))==1:
       anchor=np.where(np.array(number_of_edge)==max(number_of_edge))[0][0]
    elif np.sum(number_of_edge==max(number_of_edge))>1:
        anchor_index=number_of_node[np.where(np.array(number_of_edge)==max(number_of_edge))]
        length_anchor=[]
        order=[]
        for i in range(len(anchor_index)):
            component=[anchor_index[i]]
            k=1
            while k!=0:
                if component[-1] in np.array(Node)[:,0]:
                    index=np.where(np.array(Node)[:,0]==component[-1])[0][0]
                    component.append(np.array(Node)[index,-1])
                else:
                    k=0
            k=1
            while k!=0:
                if component[0] in np.array(Node)[:,1]:
                    index=np.where(np.array(Node)[:,1]==component[0])[0][0]
                    component.insert(0,np.array(Node)[index,0])
                else:
                    k=0
            order.append(component)
            length_anchor.append(len(component))
        components=order[np.where(np.array(length_anchor)==max(length_anchor))[0][0]]
        anchor=int(np.round(len(components)/2)-1)
    dirt_connect=[]
    remain_Node=[]
    for j in range(np.array(Node).shape[0]):
        if anchor in Node[j]:
            dirt_connect.append(Node[j])
        else:
            remain_Node.append(Node[j])
    dir_connect_list=np.array(dirt_connect).reshape(np.array(dirt_connect).shape[0]*np.array(dirt_connect).shape[1])
    dir_connect_image=dir_connect_list[np.where(dir_connect_list!=anchor)]
    pair=[]
    new_dirt_connect=[]
    k=0
    for n in range(len(dir_connect_image)):
        for N in range(np.array(remain_Node).shape[0]):
            if dir_connect_image[n] in remain_Node[N]:
                if remain_Node[N] in pair:
                    pair=pair
                else:
                    pair.append(remain_Node[N])
                k=1
        if k==0:
            new_dirt_connect.append(dirt_connect[n])
        k=0
    new_dir_connect_image=list(copy.deepcopy(dir_connect_image))
    for j in range(np.array(pair).shape[0]):
        if pair[j][0] in new_dir_connect_image:
            new_dir_connect_image.append(pair[j][1])
        elif pair[j][1] in new_dir_connect_image:
            new_dir_connect_image.append(pair[j][0])
    new_dir_connect_image.append(anchor)
    new_dir_connect_image.sort()
    new_dir_connect_image=list(set(new_dir_connect_image))
            
    middle=im_list[int(anchor)]
    warps=[]
    origins=[]
    for i in range(np.array(new_dirt_connect).shape[0]):
        index=new_dirt_connect[i]
        x=index[np.where(np.array(index)!=anchor)[0][0]]
        warped,origin=pair_(x,int(anchor),im_list)
        warps.append(warped)
        origins.append(origin)
    for j in range(np.array(pair).shape[0]):
        if pair[j][0] in dir_connect_image or pair[j][1] in dir_connect_image:
            if pair[j][0] in dir_connect_image:
                warped,origin=pair_(pair[j][1],pair[j][0],im_list)
                pano=stitich([warped,im_list[pair[j][0]]],[origin,(0,0)])
            elif pair[j][1] in dir_connect_image:
                warped,origin=pair_(pair[j][0],pair[j][1],im_list)
                pano=stitich([warped,im_list[pair[j][1]]],[origin,(0,0)])
            kp1,kp2,des1,des2,matches=feature_matching(pano,middle)
            H=find_homography_for_multi_image(kp1,kp2,matches)
            warp,origin=warp_image(pano,H)
            warps.append(warp)
            origins.append(origin)
    warps.append(middle)
    origins.append((0,0))
    final_pano=stitich(warps,origins)
    filename=filename1(new_dir_connect_image,image_list)
    cv2.imwrite('./'+out_dir+'/'+filename,final_pano)
    
def filename1(new_dir_connect_image,image_list):
    filename=(image_list[0].split('.')[0]+'_'+
                  image_list[1].split('.')[0])
    for i in range(len(new_dir_connect_image)-2):
        filename=(filename+'_'+image_list[i+2].split('.')[0])
    filename=filename+'.'+image_list[i].split('.')[1]
    return filename

if __name__ == '__main__':
    im_list,image_list=read_image(sys.argv[1])
    out_dir=sys.argv[2]
    min_keypoints=10
    Node=[] 
    if len(im_list)==2:
        pair_image(0,1,im_list,image_list,min_keypoints,Node,out_dir)
    else:
        for i in range(len(im_list)-1):
            for j in range(1+i,len(im_list),1):
                print()
                print('create pairs for image',i,j)
                Node,warp=pair_image(i,j,im_list,image_list,min_keypoints,Node,out_dir)
    if np.array(Node).shape[0]>1:
        print('Multi-image mosaic begin')
        multi_image_mosaic(Node,im_list,out_dir,image_list)
        print('Multi-image mosaic end')
    else:
        print('There are only pairs')
