import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as matimg

# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

X = 9
Y = 6

folder = 'New folder'
path = os.listdir(folder)
all_images = [os.path.join(folder+'/',i) for i in path]

ref_image = cv2.imread('advanced_straight.jpg')
ref_image_color = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

ref_image_curved = cv2.imread('advanced_color.jpg')
ref_image_curved_color = cv2.cvtColor(ref_image_curved, cv2.COLOR_BGR2RGB)

#total_images = [ref_image_color, ref_image_curved_color]
                                            
                                                ################################## Camera Calibration ##################################
        
def countBox(arg):

    point = np.zeros((Y * X, 3), np.float32)
    point[:,:2] = np.mgrid[0:X, 0:Y].T.reshape(-1,2)

    img_pts = []
    obj_pts = []

    c=0

    for i in arg:

        image = cv2.imread(i)
        C2G = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        get, corners = cv2.findChessboardCorners(C2G, (X,Y), None)

        if not get:
            print(f'Please check if the number of x and y boxes in {i} are {X} and {Y}')
            continue

        print(i, c)
        img_pts.append(corners)
        obj_pts.append(point)

        c+=1

        #corners2 = cv2.cornerSubPix(C2G, corners, (11,11), (-1,-1), criteria)

        # cv2.drawChessboardCorners(image, (9,6), corners2, get)
        # cv2.imshow('img',image)
        # cv2.waitKey()

    return img_pts, obj_pts

def calibrateCamera(img, obj, referrence_image):

    shape = referrence_image.shape[:2][::-1]
    get, mat, dist, rvec, tvec = cv2.calibrateCamera(obj, img, shape, None, None)

    undistorted = cv2.undistort(referrence_image, mat, dist, None, mat)

    return mat, dist, undistorted

img_pts, obj_pts = countBox(all_images)

# fig, ax = plt.subplots(2,2)
# fig.set_size_inches(12,6)
# ax[0,0].imshow(ref_image_color)
# ax[0,1].imshow(undistort[0])
# ax[1,0].imshow(ref_image_curved_color)
# ax[1,1].imshow(undistort[1])
# plt.show()

                                      ################################## Functions involved in pre-processing ##################################

GRADIENT = (20, 100)
B_CHANNEL = (150, 200)
L_CHANNEL = (225, 255)

def hls(img):

    Img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h = Img_hls[:,:,0]
    l = Img_hls[:,:,1]
    s = Img_hls[:,:,2]

    return h, l, s

def lab(img):

    Img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    l = Img_lab[:,:,0]
    a = Img_lab[:,:,1]
    b = Img_lab[:,:,2]

    return l, a, b

def luv(img):

    Img_luv = cv2.cvtColor(img, cv2.COLOR_BGR2Luv)
    l = Img_luv[:, :, 0]
    a = Img_luv[:, :, 1]
    b = Img_luv[:, :, 2]

    return l, a, b

def binary_lab_luv(img, bthresh, lthresh):

    l, a, b = lab(img)
    l_2, u, v = luv(img)
    binary = np.zeros_like(l)
    binary[
        ((b > bthresh[0]) & (b <= bthresh[1])) | ((l_2 > lthresh[0]) & (l_2 <= lthresh[1]))
    ] = 1

    return binary

def gradient(channel, threshold):

    sobel_gradient = cv2.Sobel(channel, cv2.CV_64F, 1, 0)
    abs_sobel = np.absolute(sobel_gradient)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    sobel_binary = np.zeros_like(scaled_sobel)
    sobel_binary[(scaled_sobel >= threshold[0]) & (scaled_sobel <= threshold[1])] = 1

    return sobel_binary

#
# h0, l0, s0 = hls(undistort)
# sbinary_hls0 = gradient(s0, GRADIENT)
# color_binary0 = np.dstack((sbinary_hls0, s_binary0, np.zeros_like(sbinary_hls0))) * 255


# fig, ax = plt.subplots(1,4)
# fig.set_size_inches(10,8)
# ax[0].imshow(undistort)
# ax[1].imshow(s_binary0, cmap='gray')
# ax[2].imshow(sbinary_hls0, cmap='gray')
# ax[3].imshow(color_binary0)
# plt.show()

TOTAL_WINDOWS = 10
MARGIN = 100
RECENTER_PIX = 50

SHAPE = ref_image_color.shape[:2][::-1]
OFFSET = 320
SRC = np.float32([
     (590, 447),
     (190, 720),
     (1200, 720),
     (685, 447)
     ])

DST = np.float32([
    [OFFSET, 0],
    [OFFSET, SHAPE[1]],
    [SHAPE[0]-OFFSET, SHAPE[1]],
    [SHAPE[0]-OFFSET, 0]
])

# copy_undistorted = undistort.copy()
# plt.figure(figsize=(10, 10))
# plt.imshow(cv2.polylines(copy_undistorted, np.int32([SRC]), True, (0,0,255), 4))
# plt.show()

M = cv2.getPerspectiveTransform(SRC, DST)
MINV = cv2.getPerspectiveTransform(DST, SRC)

mat, dist, undistort = calibrateCamera(img_pts, obj_pts, ref_image_color)
s_binary0 = binary_lab_luv(undistort, B_CHANNEL, L_CHANNEL)
warp0 = cv2.warpPerspective(undistort, M, SHAPE, flags=cv2.INTER_LINEAR)
copy_warp0 = warp0.copy()
warp0_poly = cv2.polylines(copy_warp0, np.int32([DST]), True, (0, 0, 255), 4)
warp_binary0 = cv2.warpPerspective(s_binary0, M, SHAPE, flags=cv2.INTER_LINEAR)

# plt.imshow(warp0_poly)
# plt.show()


#histogram0 = np.sum(warp_binary0[warp_binary0.shape[0]//2:,:], axis=0)

# f, ax = plt.subplots(2,2)
# f.set_size_inches(16,8)
# ax[0,0].imshow(warp_binary0, cmap='gray')
# ax[0,1].plot(histogram0)
# plt.show()

# ax[1,0].imshow(warp_binary1, cmap='gray')
# ax[1,1].plot(histogram1)
# plt.show()


def lane_pixels(warp, total_windows, margin, recenter):

    output = np.dstack((warp, warp, warp)) * 255

    histogram = np.sum(warp[warp.shape[0] // 2:,:], axis=0)

    center_point = np.int(histogram.shape[0]//2)
    left_base = np.argmax(histogram[:center_point])
    right_base = np.argmax(histogram[center_point:]) + center_point

    win_height = np.int(warp.shape[0] // total_windows)

    nonzero_pix = warp.nonzero()
    y_nonzero_pix = np.array(nonzero_pix[0])
    x_nonzero_pix = np.array(nonzero_pix[1])

    left_current = left_base
    right_current = right_base

    left_lane_indices, right_lane_indices = [], []

                               ################################## Apply sliding window to the binary transformation ##################################

    for each_window in range(total_windows):

        win_y_l = warp.shape[0] - (each_window + 1) * win_height
        win_y_h = warp.shape[0] - each_window * win_height

        win_x_left_l = left_current - margin
        win_x_left_h = left_current + margin
        win_x_right_l = right_current - margin
        win_x_right_h = right_current + margin

        cv2.rectangle(output,(win_x_left_l, win_y_l),(win_x_left_h,win_y_h),(0,255,0), 4)
        cv2.rectangle(output,(win_x_right_l, win_y_l), (win_x_right_h, win_y_h), (0,255,0), 4)

        left_indices = ((y_nonzero_pix >= win_y_l) &
                        (y_nonzero_pix < win_y_h) &
                        (x_nonzero_pix >= win_x_left_l) &
                        (x_nonzero_pix < win_x_left_h)).nonzero()[0]

        right_indices = ((y_nonzero_pix >= win_y_l) &
                         (y_nonzero_pix < win_y_h) &
                         (x_nonzero_pix >= win_x_right_l) &
                         (x_nonzero_pix < win_x_right_h)).nonzero()[0]

        #print(right_indices)
        # print((ynonzero_pix >= win_y_l))
        # print((ynonzero_pix < win_y_h))
        # print((xnonzero_pix >= win_x_right_l))
        # print((xnonzero_pix < win_x_right_h))

        left_lane_indices.append(left_indices)
        right_lane_indices.append(right_indices)

        if len(left_indices) > recenter:
            left_current = np.int(np.mean(x_nonzero_pix[left_indices]))
        if len(right_indices) > recenter:
            right_current = np.int(np.mean(x_nonzero_pix[right_indices]))

    try:
        left_lane_indices = np.concatenate(left_lane_indices)
        right_lane_indices = np.concatenate(right_lane_indices)

    except ValueError:
        print('Value Error, please check the left and right index values')

    leftx = x_nonzero_pix[left_lane_indices]
    lefty = y_nonzero_pix[left_lane_indices]
    rightx = x_nonzero_pix[right_lane_indices]
    righty = y_nonzero_pix[right_lane_indices]

    return leftx, lefty, rightx, righty, output

#a, b, c, d, e= lane_pixels(warp_binary0, TOTAL_WINDOWS, MARGIN, RECENTER_PIX)

                       ################################## Function to write lines using polynomial function ##################################

def draw_poly(leftx, lefty, rightx, righty, inp):

    fit_left = np.polyfit(lefty, leftx, 2)
    fit_right = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, inp.shape[0] - 1, inp.shape[0])

    fit_left_x = fit_left[0] * ploty ** 2 + fit_left[1] * ploty + fit_left[2]
    fit_right_x = fit_right[0] * ploty ** 2 + fit_right[1] * ploty + fit_right[2]

    return fit_left_x, fit_right_x, ploty

                       ################################## Function to apply polygon between lanes and write parameters ##################################

def draw_lane(raw_image, warp, left_fit, right_fit, ploty, m_inv, mean_curvature, vehicle_center):

    warp_copy = np.zeros_like(warp).astype(np.uint8)
    output = np.dstack((warp_copy, warp_copy, warp_copy))

    left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
    point = np.hstack((left, right))

    cv2.fillPoly(output, np.int_([point]), (0,150,255))

    warp_to_raw = cv2.warpPerspective(output, m_inv, (raw_image.shape[1], raw_image.shape[0]))
    cv2.putText(warp_to_raw, 'Radius is ' + str(round(mean_curvature, 2)) + ' m', (50, 50),
                cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(warp_to_raw, 'Vehicle center is ' + str(round(vehicle_center, 2)) + ' cm', (50, 100),
                cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

    # plt.imshow(warp_to_raw)
    # plt.show()
    view_warp_raw = cv2.addWeighted(raw_image, 1, warp_to_raw, 0.3, 0)

    return view_warp_raw


X_PIXEL_METER = 3.7/700
Y_PIXEL_METER = 30/720

def search_poly(inp, raw, margin):

                            ################################## Draw line over detected lane using polynomial function ##################################
    
    nonzero = inp.nonzero()
    ynonzero_pix = np.array(nonzero[0])
    xnonzero_pix = np.array(nonzero[1])

    leftx, lefty, rightx, righty, output = lane_pixels(inp, TOTAL_WINDOWS, MARGIN, RECENTER_PIX)

    if ((len(leftx)==0) or (len(lefty)==0) or (len(rightx)==0) or (len(righty)==0)):

        right_curvature = 0
        left_curvature = 0
        out = np.dstack((inp,inp,inp)) * 255

    else:

        fit_left = np.polyfit(lefty, leftx, 2)
        fit_right = np.polyfit(righty, rightx, 2)

        left_lane_indices = ((xnonzero_pix > (fit_left[0] * ynonzero_pix ** 2 + fit_left[1] * ynonzero_pix + fit_left[2] - margin)) &
                             (xnonzero_pix < (fit_left[0] * ynonzero_pix ** 2 + fit_left[1] * ynonzero_pix + fit_left[2] + margin)))

        right_lane_indices = ((xnonzero_pix > (fit_right[0] * ynonzero_pix ** 2 + fit_right[1] * ynonzero_pix + fit_right[2] - margin)) &
                             (xnonzero_pix < (fit_right[0] * ynonzero_pix ** 2 + fit_right[1] * ynonzero_pix + fit_right[2] + margin)))

        leftx = xnonzero_pix[left_lane_indices]
        lefty = ynonzero_pix[left_lane_indices]
        rightx = xnonzero_pix[right_lane_indices]
        righty = ynonzero_pix[right_lane_indices]

        fit_left_x, fit_right_x, ploty = draw_poly(leftx, lefty, rightx, righty, inp)

        # output[ynonzero_pix[left_lane_indices], xnonzero_pix[left_lane_indices]] = [255, 0, 0]
        # output[ynonzero_pix[right_lane_indices], xnonzero_pix[right_lane_indices]] = [0, 0, 255]

#search_poly(warp_binary1, 50)

                           ################################## Obtain radius of curvature of the lane and vehicle position ##################################
    
        y_top = np.max(ploty)

        right_curvature = np.polyfit(ploty * Y_PIXEL_METER, fit_right_x * X_PIXEL_METER, 2)
        left_curvature = np.polyfit(ploty * Y_PIXEL_METER, fit_left_x * X_PIXEL_METER, 2)

        left_radius_curvature = ((1 + (2 * left_curvature[0] * y_top * Y_PIXEL_METER + left_curvature[1]) ** 2) ** 1.5)\
                                / np.absolute( 2 * left_curvature[0])

        right_radius_curvature = ((1 + (2 * right_curvature[0] * y_top * Y_PIXEL_METER + right_curvature[1]) ** 2) ** 1.5)\
                                / np.absolute( 2 * right_curvature[0])

        mean_curvature = (left_radius_curvature + right_radius_curvature) / 2

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        raw_shape = raw.shape
        vehicle_position = raw_shape[1]/2
        camera_center = raw_shape[0]

        ground_left = left_fit[0] * camera_center ** 2 + left_fit[1] * camera_center + left_fit[2]
        ground_right = right_fit[0] * camera_center ** 2 + right_fit[1] * camera_center + right_fit[2]

        vehicle_center = (np.abs(vehicle_position - mean_curvature) * X_PIXEL_METER) / 10

        output = np.dstack((inp, inp, inp)) * 255
        window = np.zeros_like(output)

        output[ynonzero_pix[left_lane_indices], xnonzero_pix[left_lane_indices]] = [255, 0, 0]
        output[ynonzero_pix[right_lane_indices], xnonzero_pix[right_lane_indices]] = [0, 0, 255]

        left_1 = np.array([np.transpose(np.vstack([fit_left_x-margin, ploty]))])
        left_2 = np.array([np.flipud(np.transpose(np.vstack([fit_left_x+margin, ploty])))])
        left = np.hstack((left_1, left_2))

        right_1 = np.array([np.transpose(np.vstack([fit_right_x-margin, ploty]))])
        right_2 = np.array([np.flipud(np.transpose(np.vstack([fit_right_x+margin, ploty])))])
        right = np.hstack((right_1, right_2))

        cv2.fillPoly(window, np.int_([left]), (0,200,0))
        cv2.fillPoly(window, np.int_([right]), (0,200,0))

        margin_image = cv2.addWeighted(output, 1, window, 0.3, 0)

        # f, ax = plt.subplots(1, 2)
        # f.set_size_inches(16, 8)
        # ax[0].imshow(margin_image)
        # ax[1].imshow(window)
        # ax[1].plot(fit_left_x, ploty, color='yellow')
        # ax[1].plot(fit_right_x, ploty, color='yellow')
        # plt.show()

        warp_to_raw = draw_lane(raw, inp, fit_left_x, fit_right_x, ploty, MINV, mean_curvature, vehicle_center)

        return warp_to_raw

# warp_to_raw = search_poly(warp_binary0, ref_image_color, 50)
# plt.imshow(warp_to_raw)
# plt.show()

capture = cv2.VideoCapture('project_video.mp4')

while capture.isOpened():

    get, frame = capture.read()
    if get == True:

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mat, dist, undistort = calibrateCamera(img_pts, obj_pts, frame)
        s_binary0 = binary_lab_luv(undistort, B_CHANNEL, L_CHANNEL)

        warp0 = cv2.warpPerspective(undistort, M, SHAPE, flags=cv2.INTER_LINEAR)
        copy_warp0 = warp0.copy()

        warp_binary0 = cv2.warpPerspective(s_binary0, M, SHAPE, flags=cv2.INTER_LINEAR)
        warp_to_raw = search_poly(warp_binary0, frame, 50)
        warp_to_raw = cv2.cvtColor(warp_to_raw, cv2.COLOR_RGB2BGR)

        cv2.imshow('frame',warp_to_raw)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

capture.release()
cv2.destroyAllWindows()






