import numpy as np
import cv2
import os
from sklearn.cluster import MeanShift, estimate_bandwidth
import sys

'''load image pairs as BRG and grayscale'''
def load_image(i):
    im_list=os.listdir('./hw7data')
    old_frame=cv2.imread('./hw7data/'+im_list[i])
    old_gray=cv2.imread('./hw7data/'+im_list[i],cv2.IMREAD_GRAYSCALE)
    frame=cv2.imread('./hw7data/'+im_list[i+1])
    frame_gray=cv2.imread('./hw7data/'+im_list[i+1],cv2.IMREAD_GRAYSCALE)
    return old_frame,old_gray,frame,frame_gray

'''shi-Tomasi corners detect'''
def corners(img):
    feature_params = dict( maxCorners = 300,qualityLevel = 0.4,minDistance = 3,blockSize = 10 )
    p0=cv2.goodFeaturesToTrack(img,**feature_params)
    return p0

'''ORB corners detect'''
def orb(img):
    num_features = 300
    orb = cv2.ORB_create(num_features)
    kp, des = orb.detectAndCompute(img, None)
    p0=np.array([[kp[i].pt[0],kp[i].pt[1]] for i in range(len(kp))]).astype(np.float32)
    p0=p0.reshape(-1,1,2)
    return p0

'''L-K to find the matched flow vectors'''
def get_keypoint(old_gray,frame_gray,p0):
    lk_params = dict( winSize  = (10,10),maxLevel = 2,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    good_new = p1[st==1]
    good_old = p0[st==1]
    return good_new,good_old

'''calculate the FOE and draw it on the image'''
def draw_FOE(frame,good_new,good_old,m):
    color= (0,255,0)
    for i,(new,old) in enumerate(zip(good_new,good_old)): 
        a,b = new.ravel()        
        c,d = old.ravel()       
        cv2.arrowedLine(frame, (a,b),(c,d), color, 2)     
    norm_=np.linalg.norm(good_new-good_old,axis=1)
    if np.sum(norm_>1)/len(norm_)>0.5:
        print('Most motion vectors are moving',file=f)
        print('camera is moving',file=f)
        F,mask=cv2.findFundamentalMat(good_new, good_old, cv2.FM_RANSAC)
        H, H_mask = cv2.findHomography(good_new, good_old, cv2.RANSAC,5.0)
        bad_new = good_new[H_mask.ravel()==0]
        bad_old = good_old[H_mask.ravel()==0]
        good_old = good_old[mask.ravel()==1]
        good_new = good_new[mask.ravel()==1]
        flow_vector=good_old-good_new
        b=good_old[:,0]*flow_vector[:,1]-good_old[:,1]*flow_vector[:,0]
        A=np.array([flow_vector[:,1],flow_vector[:,0]]).T
        FOE=abs(np.dot(np.dot(np.linalg.inv(np.dot(A.T,A)),A.T),b))
        red=(0,0,255)
        cv2.circle(frame,(int(FOE[0]),int(FOE[1])),7,red,-1)
    else:
        print('Above 50%  motion vectors are stay in original position',file=f)
        print('camera is stationary',file=f)
        H, H_mask = cv2.findHomography(good_new, good_old, cv2.RANSAC,5.0)
        if np.sum(H_mask.ravel()==0)>15:
            bad_new = good_new[H_mask.ravel()==0]
            bad_old = good_old[H_mask.ravel()==0]
        else:
            new_norm=np.linalg.norm(good_new-good_old,axis=1)
            bad_new=good_new[new_norm==max(new_norm)]
            bad_old=good_old[new_norm==max(new_norm)]
    cv2.imwrite('./'+'FOE_'+'{}'.format(m)+'.jpg',frame)
    return bad_new,bad_old

'''independent object detection and meanshift clustering'''
def independent_object_detection(bad_new,bad_old,frame,m):
    if len(bad_new)>15:
        bandwidth = estimate_bandwidth(bad_new, quantile=0.2, n_samples=-1)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(bad_new)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        print('cluster_centers:',cluster_centers)
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        print("number of estimated clusters : %d" % n_clusters_)
        for i in range(n_clusters_):
            count=np.sum(labels==i)
            if count>4:
                new_bad=bad_new[labels==i]
                old_bad=bad_old[labels==i]
                color=np.random.randint(0,255,size=(3))
                for j,(new,old) in enumerate(zip(new_bad,old_bad)): 
                    a,b = new.ravel()        
                    c,d = old.ravel()       
                    cv2.arrowedLine(frame, (c,d),(a,b),(int(color[0]),int(color[1]),int(color[2])), 2)
                xmin=int(cluster_centers[i][0]-40)
                xmax=int(cluster_centers[i][0]+40)
                ymin=int(cluster_centers[i][1]-40)
                ymax=int(cluster_centers[i][1]+40)
                cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),(0,255,0),3)
    else:
        color=np.random.randint(0,255,size=(3))
        for i in range(len(bad_new)):
            a,b = bad_new[i].ravel()        
            c,d = bad_old[i].ravel() 
            cv2.arrowedLine(frame, (c,d),(a,b),(int(color[0]),int(color[1]),int(color[2])), 2)
            xmin=int(bad_new[0][0]-40)
            xmax=int(bad_new[0][0]+40)
            ymin=int(bad_new[0][1]-40)
            ymax=int(bad_new[0][1]+40)
            cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),(0,255,0),3)
    cv2.imwrite('./'+'detect_'+'{}'.format(m)+'.jpg',frame)
    
if __name__ == '__main__':
    f=open('./statements.txt', "w")
    i=sys.argv[1]
    print('pairs:',i,i+1,file=f)
    old_frame,old_gray,frame,frame_gray=load_image(i)
    p0=corners(old_gray)
    #p0=orb(old_gray)
    good_new,good_old=get_keypoint(old_gray,frame_gray,p0)
    bad_new,bad_old=draw_FOE(old_frame,good_new,good_old,i)
    independent_object_detection(bad_new,bad_old,frame,i)
    f.close()