import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt 

# function to obtain histogram of an image 
def ICV_hist(img): 
      
    m = img.shape[0]
    n = img.shape[1]
    # empty list to store the count  
    # of each intensity value 
    count = np.empty((3,256)) 
      
    # loop to traverse each intensity value
    for c in range(3):

        for k in range(0, 256):  
            count1 = 0
          
            # loops to traverse each pixel in the image  
            for i in range(m): 
                for j in range(n): 
                    if img[i,j,c]== k: 
                        count1+= 1
            count[c,k] = count1 
          
    return count

# Intersection of two histograms
def ICV_Intersection(count1,count2,size):

    intersection = np.zeros((3))
    intersection_normalized = np.zeros((3))
    # Interection histogram values
    diff = np.zeros((3,256))
    diff_n = np.zeros((3,256))
    for c in range (3):
        for i in range(0,256):
            intersection[c] += min(count1[c,i],count2[c,i])
            diff[c,i] = min(count1[c,i],count2[c,i])
        # Normalizing the histogram

        intersection_normalized[c] = intersection[c]/size
        diff_n[c] = diff[c]/size 
    return intersection, intersection_normalized, diff, diff_n

def ICV_plot(count,image):
    # plot the histogram
    
    fig = plt.figure()

    # show original image
    fig.add_subplot(232)
    plt.title(' image')
    plt.axis('off')
    plt.imshow(image)

    # plot the histogram
    fig.add_subplot(234)
    plt.title('histogram Blue')
    plt.plot((count[0]),'b-')

    fig.add_subplot(235)
    plt.title('histogram green')
    plt.plot((count[1]),'g-')

    fig.add_subplot(236)
    plt.title('histogram Red')
    plt.plot((count[2]),'r-')
    
    
    plt.show() 

def ICV_plt(count):
    # plot the intersection histogram
    
    fig = plt.figure()
    
    fig.add_subplot(234)
    plt.title('Intesection histogram Blue')
    plt.plot((count[0]),'b-')

    fig.add_subplot(235)
    plt.title('Intesection histogram green')
    plt.plot((count[1]),'g-')

    fig.add_subplot(236)
    plt.title('Intesection histogram Red')
    plt.plot((count[2]),'r-')
    
    
    plt.show() 

# To plot histogram of the video
def ICV_vid():
    # video file
    cap = cv.VideoCapture('Dataset/DatasetB.avi')    
    while(cap.isOpened()):

        ret, frame = cap.read()
        if ret == True:
            # Creating the histogram of the frame
            hist = ICV_hist(frame)
            ICV_plt(hist)
            # Comment the break if needed for the whole video
            break
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()


def ICV_vidIntersection():
    # Video file
    cap = cv.VideoCapture('Dataset/DatasetB.avi')
    ret, frame1 = cap.read()
    size = frame1.shape[0]*frame1.shape[1]
    hist1 = ICV_hist(frame1)    
    while(cap.isOpened()):

        ret, frame2 = cap.read()
        if ret == True:
            hist2 = ICV_hist(frame2)
            # Calulating the histogram of two consequtive frames
            # intersection and intersection_normalized are the values of the intersection and normalized intersection respectively.
            # diff and diff_n are the histogram values of the intersection and the normalized intersection values respectively.
            intersection, intersection_normalized, diff, diff_n = ICV_Intersection(hist1,hist2,size)
            # Plotting the I(t-1) frame
            ICV_plot(hist1, frame1)
            # Plotting the It frame
            ICV_plot(hist2, frame2)
            # Plotting the intersection histogram
            print("------------Plotting the intersection of the histograms-------------")
            ICV_plt(diff)
            # Plotting the normalized histogram intersection
            print("------------Plotting the normalized intersection of the histograms-------------")
            ICV_plt(diff_n)
            print("The intersection values of each channel: ", intersection)
            print("The normalized intersection values of each channel: ", intersection_normalized)
            hist1 = hist2
            # Comment the break if needed for the whole video
            break
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()



print("----------------Histograms-----------------------")

#------ Histogram of two non consecutive frames-------------
cap = cv.VideoCapture('Dataset/DatasetB.avi')
# Taking the first frame
cap.set(cv.CAP_PROP_POS_FRAMES, 0)
ret, frame1 = cap.read()
# Taking the frame in the middle
num = int(cap.get(cv.CAP_PROP_FRAME_COUNT)/2)
cap.set(cv.CAP_PROP_POS_FRAMES, num)
ret, frame2 = cap.read()

# Plotting histogram of first frame
print("-------Plotting histogram of the first frame--------")
hist1 = ICV_hist(frame1)
ICV_plot(hist1,frame1)

# Plotting histogram of the middle frame
print("-------Plotting histogram of the middle frame--------")
hist2 = ICV_hist(frame2)
ICV_plot(hist2,frame2)

# ----------Calculating the intersection of 2 consecutive frames in a histogram---------------
print("-------Taking the intersection of consequtive frames--------")
ICV_vidIntersection()
