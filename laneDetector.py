import FiraAuto
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt


toothFilterPulseWidth = 70

def warp(img):
    pts1 = np.float32([[59,145], [210,145], [8,195], [255,195]])
    pts2 = np.float32([[0,0], [250,0], [0,350], [250,350]])
    # pts1 = np.float32([[1,167], [75,112], [245,167], [180,112]])
    # pts2 = np.float32([[0,350], [0,0], [250,350], [250,0]])
    # pts1 = np.float32([[1,167], [75,112], [245,167], [180,112]])
    # pts2 = np.float32([[0,350], [0,0], [250,350], [250,0]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped_img = cv2.warpPerspective(img, matrix, (250,350))
    return warped_img

def unwarp(img):
    pts1 = np.float32([[59,145], [210,145], [8,195], [255,195]])
    pts2 = np.float32([[0,0], [250,0], [0,350], [250,350]])
    # pts1 = np.float32([[1,167], [75,112], [245,167], [180,112]])
    # pts2 = np.float32([[0,350], [0,0], [250,350], [250,0]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    unwarped_img = cv2.warpPerspective(img, matrix, (256,256), flags=cv2.WARP_INVERSE_MAP)
    return unwarped_img


def colorThreshold(warped_img):
    channel = warped_img[:, :, 0]
    threshold = (170, 255)
    output = np.zeros_like(channel)
    output[(channel>=threshold[0]) & (channel<=threshold[1])]=255
    return output


def histogram(binary_img):
    image = binary_img/255
    buttom_half = image[ image.shape[0]//2:, : ]
    hist = np.sum( buttom_half, axis=0)
    return hist

def createZeroSignal(width=500, value=0):
    return np.ones(width) * value

def createToothFilter(pulseWidth=50, pulseHeight=150):
    out = np.zeros(pulseWidth)
    for i in range(pulseWidth//2+1):
        out[i] = i * (pulseHeight / pulseWidth)
        out[-i] = i * (pulseHeight / pulseWidth)
    return out

def createConvFileter(width=250, c1=89, c2=139, bias=0, value=0):
    signal = createZeroSignal(width, value=value)
    signal[
    c1 - toothFilterPulseWidth // 2 + bias: c1 + 70 // 2 + bias] += createToothFilter()
    signal[
    c2 - toothFilterPulseWidth // 2 + bias: c2 + 70 // 2 + bias] += createToothFilter()
    # print('centers:', c1, c2)
    return signal

def convolve(hist, fullLenght=0, midLength=0, filter=None):  # TODO what does midlen do?
    filter = createConvFileter() if filter is None else filter
    return np.convolve(filter, hist)#[midLength // 2:2 * fullLenght - (midLength // 2)]


def oneOrTwo(binary_img):
    hist = histogram(binary_img)
    midpoint = np.int32(hist.shape[0]//2)
    leftx_base = np.argmax(hist[:midpoint])
    rightx_base = np.argmax(hist[midpoint:]) + midpoint

    print('left=',leftx_base)
    print('right=',rightx_base)
    if ((hist[rightx_base]<90) | (hist[leftx_base]<90) | (rightx_base-leftx_base<40)):
        line = 1
    else:
        line = 2
    print('line=',line)
    return line


def findLanePixels2(binary_img):
    img_copy = binary_img.copy()
    hist = histogram(binary_img)
    midpoint = np.int32(hist.shape[0]//2)
    left_base = np.argmax(hist[:midpoint])
    right_base = np.argmax(hist[midpoint:]) + midpoint
    # print('left b=',hist[left_base])
    # print('right b=',hist[right_base])

    leftx_current = left_base
    rightx_current = right_base

    min_pixels = 10
    margin = 25
    nwindows = 8
    window_height =np.int32(binary_img.shape[0]//nwindows)

    nonzero = binary_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = []
    right_lane_inds = []
    left_rec, right_rec = 1, 1
    for window in range (nwindows):
        win_y_low = binary_img.shape[0] - ((window+1)*window_height)
        win_y_high = binary_img.shape[0] - ( window*window_height)
        if (leftx_current-margin<0):
            win_xleft_low=0
        else:    
            win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        if (rightx_current+margin>binary_img.shape[1]):
            win_xright_high = binary_img.shape[1]
        else:     
            win_xright_high = rightx_current + margin

        if (left_rec != 0):
            cv2.rectangle( binary_img, (win_xleft_low , win_y_low), (win_xleft_high , win_y_high), (255,0,0), 2)
        if (right_rec != 0):
            cv2.rectangle( binary_img, (win_xright_low , win_y_low), (win_xright_high , win_y_high), (255,0,0), 2)


        good_left_inds = ((nonzeroy>=win_y_low) & (nonzeroy<win_y_high) & (nonzerox>=win_xleft_low) & (nonzerox<win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy>=win_y_low) & (nonzeroy<win_y_high) & (nonzerox>=win_xright_low) & (nonzerox<win_xright_high)).nonzero()[0]

        if ((len(good_left_inds) > min_pixels) & (len(good_right_inds) > min_pixels)):
            leftx_current = np.int32( np.mean( nonzerox[good_left_inds]))
            left_lane_inds.append(good_left_inds)
            rightx_current = np.int32( np.mean( nonzerox[good_right_inds]))
            right_lane_inds.append(good_right_inds)
            # print('1')
        elif ((len(good_left_inds) > min_pixels) & (len(good_right_inds) < min_pixels)):  
            leftx_current = np.int32( np.mean( nonzerox[good_left_inds]))
            left_lane_inds.append(good_left_inds)
            right_rec = 0
            # print('2')
        elif ((len(good_left_inds) < min_pixels) & (len(good_right_inds) > min_pixels)):   
            rightx_current = np.int32( np.mean( nonzerox[good_right_inds]))
            right_lane_inds.append(good_right_inds) 
            left_rec = 0
            # print('3')    
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        pass    
   
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    out_img = np.dstack((binary_img, binary_img, binary_img))
    
    return leftx, lefty, rightx, righty, out_img


def findLanePixels1(binary_img):
    hist = histogram(binary_img)
    base = np.argmax(hist)
    midpoint = binary_img.shape[1]//2
    if (base<midpoint):
        movepix = np.int32(90)
    else:
        movepix = np.int32(-90)
    nwindows = 8
    margin = 20
    minpix = 30

    window_height = np.int32(binary_img.shape[0]//nwindows)
    x_current = base

    nonzero = binary_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    lane_inds = []
    rec = 1
    for window in range (nwindows):
        win_y_low = binary_img.shape[0] - ((window+1)*window_height)
        win_y_high = binary_img.shape[0] - ( window*window_height)
        win_x_low = x_current - margin
        win_x_high = x_current + margin
        
        if (rec != 0):
            cv2.rectangle( binary_img, (win_x_low , win_y_low), (win_x_high , win_y_high), (255,0,0), 2)

        good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
        
        if ((len(good_inds) > minpix)):
            x_current = np.int32( np.mean( nonzerox[good_inds]))
            lane_inds.append(good_inds)
        else:
            rec = 0

    try:
        lane_inds = np.concatenate(lane_inds)
    except ValueError:
        pass
    
    linex = nonzerox[lane_inds]
    liney = nonzeroy[lane_inds]
    out_img = np.dstack((binary_img, binary_img, binary_img))
    return linex, liney, out_img, x_current, movepix


def fitPoly2(binary_warped_img):
    leftx, lefty, rightx, righty,out_img = findLanePixels2(binary_warped_img)
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, binary_warped_img.shape[0]-1, binary_warped_img.shape[0])
    
    try:
        leftx_fit = left_fit[0]*ploty**2+left_fit[1]*ploty+left_fit[2]
        rightx_fit = right_fit[0]*ploty**2+right_fit[1]*ploty+right_fit[2]
    except TypeError:
        leftx_fit = ploty**2+ploty
        rightx_fit = ploty**2+ploty
    # print(left_fit)
    
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx]= [0, 0, 255]
    mid_fit = [(left_fit[0]+right_fit[0])/2, (left_fit[1]+right_fit[1])/2, (left_fit[2]+right_fit[2])/2]
    midx_fit = mid_fit[0]*ploty**2 + mid_fit[1]*ploty + mid_fit[2]
    mid_curve = np.column_stack((midx_fit.astype(np.int32), ploty.astype(np.int32)))
    cv2.polylines(out_img, [mid_curve], False, (0,0,255))
    left_curve = np.column_stack((leftx_fit.astype(np.int32), ploty.astype(np.int32)))
    right_curve = np.column_stack((rightx_fit.astype(np.int32), ploty.astype(np.int32)))
    cv2.polylines(out_img, [left_curve, right_curve], False, (0,255,255))
    # plt.plot(leftx_fit, ploty, color='yellow')
    return mid_fit, out_img, left_fit, right_fit


def fitPoly1 (binary_img):
    linex, liney, out_img, x_current, movepix= findLanePixels1(binary_img)

    try:
        first_fit = np.polyfit( liney, linex, 2)
        # second_fit = np.array([first_fit[0], first_fit[1], first_fit[2]+movepix])
        mid_fit = np.array([first_fit[0], first_fit[1], first_fit[2]+movepix])
    except TypeError:
        print('yoooooooooooooooooooo')
        first_fit = np.array([5,8,1])
        # second_fit = np.array([5,8,1+movepix])
        mid_fit = np.array([5,8,1+movepix])
        pass

    ploty = np.linspace(0, binary_img.shape[0]-1, binary_img.shape[0])

    try:
        first_fitx = first_fit[0]*ploty**2 + first_fit[1]*ploty + first_fit[2]
        # second_fitx = second_fit[0]*ploty**2 + second_fit[1]*ploty + second_fit[2]
        mid_fitx = mid_fit[0]*ploty**2 + mid_fit[1]*ploty + mid_fit[2]
    except TypeError:
        print('Could\'nt find the polynomial')
        first_fitx = 1*ploty**2 + 1*ploty
        mid_fitx = 1*ploty**2 + 1*ploty
        # second_fitx = 1*ploty**2 + 1*ploty
    # print(movepix)
    # print(first_fit)
    # print(second_fit)
    out_img[liney, linex] = [255, 0, 0]
    # out_img[righty, rightx] = [0, 0, 255]
    # mid_fit = [(first_fit[0]+second_fit[0])/2, (first_fit[1]+second_fit[1])/2, (first_fit[2]+second_fit[2])/2]
    mid_fitx = mid_fit[0]*ploty**2 + mid_fit[1]*ploty + mid_fit[2]
    mid_curve = np.column_stack((mid_fitx.astype(np.int32), ploty.astype(np.int32)))
    cv2.polylines(out_img, [mid_curve], False, (0,0,255))
    first_curve = np.column_stack((first_fitx.astype(np.int32), ploty.astype(np.int32)))
    # second_curve = np.column_stack((second_fitx.astype(np.int32), ploty.astype(np.int32)))
    cv2.polylines(out_img, [first_curve], False, (0,255,255))
    # cv2.imshow('out img', out_img)
    # cv2.waitKey(0)
    return mid_fit, out_img


def calculate_pd(warped_binary_img):
    laneNumber = oneOrTwo(warped_binary_img)
    if laneNumber ==1:
        mid_fit, out_img=fitPoly1(warped_binary_img)
    else:
        mid_fit, out_img, left_fit, right_fit = fitPoly2(warped_binary_img)
    midpoint_img = np.int32(warped_binary_img.shape[1]//2)
    midpoint_poly = np.polyval(mid_fit, warped_binary_img.shape[0]-5)
    p = midpoint_img - midpoint_poly

    point1 = [warped_binary_img.shape[0]-5, midpoint_poly]
    point2_x = np.polyval(mid_fit, warped_binary_img.shape[0]-50)
    point2 = [warped_binary_img.shape[0]-50, point2_x]
    mid_derivative = (point1[1]-point2[1])/(point1[0]-point2[0])
    d = np.arctan([mid_derivative])*180/np.pi
    return p, d, out_img, mid_fit


def base_finder(binary_warped_img):
    zeroImage = np.zeros_like(binary_warped_img)
    bases = []
    hist = histogram(binary_warped_img)
    convolutionInitialResults = convolve(hist, 250, 250)  # TODO
    convArgMax = np.argmax(convolutionInitialResults)
    convMaxValue = convolutionInitialResults[convArgMax]
    print('convArgMax: ', convArgMax)
    cv2.circle(zeroImage, (convArgMax, 320), 3, 255, 1)
    return zeroImage


def fill_two_line(img,line_1,line_2):
    ploty_bias = np.linspace(0,350,351)
    line_1 = np.poly1d(line_1)
    line_2 = np.poly1d(line_2)
    line_first = np.column_stack((line_1(ploty_bias).astype(np.int32), ploty_bias.astype(np.int32)))  # for demonstration
    line_second = np.column_stack((line_2(ploty_bias).astype(np.int32),ploty_bias.astype(np.int32)))
    clone = np.zeros_like(img)
    # line_first = np.array([np.transpose(np.vstack([line_1(ploty_bias), ploty_bias]))])
    # line_second = np.array(np.flipud(np.transpose(np.vstack([line_2(ploty_bias), ploty_bias]))))
    points=np.vstack((line_first, np.flip(line_second,0)))
    print(points.shape)
    cv2.fillPoly(clone, [points], (0, 255, 0))
    clone = cv2.addWeighted(img, 1, clone, 1, 0)
    return clone


def arrowline_drawer(img,mean_fit):
    mean_fit = np.poly1d(mean_fit)
    # Start coordinate, here (225, 0)
    # represents the top right corner of image
    start_points = [0, 60 , 120 , 180 , 240 , 300 ]
    # start_points.reverse()
    end_points = [35 , 95 , 155  , 215  , 275 , 335]
    # end_points.reverse()
    for j,jj in zip(start_points,end_points):
        # print("hi", j , jj)
        start_point = (int(mean_fit(jj)) , jj)
        # print(start_point)
        # End coordinate
        end_point = (int(mean_fit(j)) , j)
        
        # Red color in BGR
        color = (31, 95 , 250)
        # Line thickness of 9 px
        thickness = 5
        # Using cv2.arrowedLine() method
        # Draw a red arrow line
        # with thickness of 9 px and tipLength = 0.5
        img = cv2.arrowedLine(img, start_point, end_point, color, thickness,tipLength = 0.5)

    return img


car = FiraAuto.car()

#connecting to the server (Simulator)
car.connect("127.0.0.1", 25001)

#Counter variable
counter = 0

debug_mode = False
#sleep for 3 seconds to make sure that client connected to the simulator 
time.sleep(3)
n=2
try:
    while(True):
        #Counting the loops
        
        counter = counter + 1

        #Start getting image and sensor data after 4 loops. for unclear some reason it's really important 
        if(counter > 4):
            #returns a list with three items which the 1st one is Left sensor data, the 2nd one is the Middle Sensor data, and the 3rd is the Right one.
            sensors = car.getSensors() 
            #EX) sensors[0] returns an int for left sensor data in cm

            #returns an opencv image type array. if you use PIL you need to invert the color channels.
            dst = car.getImage()
            #returns an integer which is the real time car speed in KMH
            carSpeed = car.getSpeed()

            #showing the opencv type image
            # rgb_img = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
            warped_img = warp(dst)
            warped_rgb_img = cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)
            binary_img = colorThreshold(warped_rgb_img)
            p, d, out_img, mid_fit = calculate_pd(binary_img)
            print('p:', p)
            print('d:', d)
            clone = arrowline_drawer(warped_img, mid_fit)
            
            unwarped = unwarp(clone)
            output_img = cv2.bitwise_or(unwarped, dst)
            cv2.imshow('frames', output_img)
            # warped_img=  cv2.resize(warped_img,(warped_img.shape[1]*2,warped_img.shape[0]*2))
            cv2.imshow('warped', out_img)

            if cv2.waitKey(1) == ord("s"):
                name = "pic"+str(n)+".jpg"
                n+=1
                cv2.imwrite(name,dst)
            elif cv2.waitKey(1) == ord("q"):
                break
            # if cv2.waitKey(10) == ord('q'):
            #     break
            time.sleep(0.001)
        #A brief sleep to make sure everything 
        # plt.show()
        #Set the power of the engine the car to 20, Negative number for reverse move, Range [-100,100]
        car.setSpeed(100)

        #Get the data. Need to call it every time getting image and sensor data
        car.getData()

        
finally:
    car.stop()