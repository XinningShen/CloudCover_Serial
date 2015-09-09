import cv2
import numpy as np
from matplotlib.pyplot import *
from datetime import timedelta
from scipy.optimize import fmin
from scipy.optimize import fsolve
import glob

HEIGHT = 640
WIDTH = 640


def getCannyPic(filename):
    """
    Canny Edge Detection on each image (using bilateralFilter in OpenCV lib)
    """
    img_original_bgr = cv2.imread(filename)
    gray = cv2.cvtColor(img_original_bgr, cv2.COLOR_BGR2GRAY)

    gray_bilateralFilter = cv2.bilateralFilter(gray, 5, 50, 50)

    dst = cv2.Canny(gray_bilateralFilter, 100, 200)

    return dst


def getRGBColorPic(filename):
    """
    RGB Color Detection on each image (using 4 criteria currently)
    """
    binaryPic = np.zeros((HEIGHT, WIDTH), np.uint8)
    img_original = cv2.imread(filename)

    for x in range(0, WIDTH):
        for y in range(0, HEIGHT):
            b = img_original.item(y, x, 0)
            g = img_original.item(y, x, 1)
            r = img_original.item(y, x, 2)
            criterion_1 = (b > 120 and g > 100 and r < 100 and (g - r) > 20)
            criterion_2 = (b > 100 and g > 100 and r > 100)
            criterion_3 = (b > 100 and g < 100 and r < 100 and (g - r) > 20)
            criterion_4 = (b < 100 and g < 100 and r < 100 and (b - g) > 20 and (g - r) > 20)
            if criterion_1 or criterion_2 or criterion_3 or criterion_4:
                binaryPic[y, x] = 1     # Sky Region
            else:
                binaryPic[y, x] = 0     # Non-sky Region
    return binaryPic


def getResultPic(result, colorCountPic, edgeCountPic, color_threshold, edge_threshold):
    """
    Generate gray image with the combination of rgb color and canny edge counting info. Set threshold for color and edge seperately.
    """
    for x in range(0, WIDTH):
        for y in range(0, HEIGHT):
            if colorCountPic[y, x] >= color_threshold and edgeCountPic[y, x] < edge_threshold:
                result[y, x] = 255  # Sky Region
            else:
                result[y, x] = 0    # Non-sky region


def getCorrelationCoefficient(result, start, end, step, files, mask):
    """
    Calculate (red component - green component) as correlation coefficient of each disjoint region over all the image stream. 
    """
    image_num = (end - start) / step + 1

    contourID = [[0 for i in range(image_num)] for j in range(3)]

    contour0_current = 0
    contour1_current = 0
    contour2_current = 0
    count = 0

    for index in range(start, end, step):
        fileName = files[index]
        print fileName
        img = cv2.imread(fileName)

        for x in range(0, HEIGHT):
            for y in range(0, WIDTH):
                grey_level = mask[x, y] / 80
                b = np.int16(img.item(x, y, 0))
                g = np.int16(img.item(x, y, 1))
                r = np.int16(img.item(x, y, 2))
                if grey_level == 1:
                    contour0_current += np.absolute(r - b)
                elif grey_level == 2:
                    contour1_current += np.absolute(r - b)
                elif grey_level == 3:
                    contour2_current += np.absolute(r - b)
        contourID[0][count] = contour0_current
        contourID[1][count] = contour1_current
        contourID[2][count] = contour2_current
        contour0_current = 0
        contour1_current = 0
        contour2_current = 0
        count += 1

    print contourID
    temp_list = []
    for i in range(1, len(result)):
        temp_list.append(contourID[0])
        temp_list.append(contourID[i])
        print 'temp_list = ', temp_list
        cc_matrix = np.corrcoef(temp_list)  # get correlation coefficient
        print 'cc_matrix', cc_matrix
        if cc_matrix[0, 1] >= 0.3:  # if corr coeff is larger than 0.3, we think it is relevant, i.e. real sky region
            result[i - 1] = True

    print result


def getSkyRegion(path):
    """
    Generate Sky Region Mask using rgb color info and canny edge detection
    """
    pixelEdgeCount = np.zeros((HEIGHT, WIDTH), np.uint8)
    pixelColorCount = np.zeros((HEIGHT, WIDTH), np.uint8)

    resultPic = np.zeros((HEIGHT, WIDTH), np.uint8)
    images_count = 0

    # Generate image-stream list (roughly 200 pics), pick image from 1/5 to 4/5 of image-stream every 10 pics.
    images = glob.glob(path + '/*.jpg')
    dir_len = len(images)
    start_pos = dir_len / 5
    end_pos = dir_len * 4 / 5
    step = (end_pos - start_pos) / 10

    for index in range(start_pos, end_pos, step):
        fileName = images[index]
        print fileName
        currentCannyPic = np.zeros((HEIGHT, WIDTH), np.uint8)
        currentCannyPic = getCannyPic(fileName)     # Get Canny Edge Pic
        currentCannyPic[currentCannyPic < 127] = 0
        currentCannyPic[currentCannyPic >= 127] = 1 # Convert Canny Edge Pic to Binary Pic

        pixelEdgeCount = np.add(pixelEdgeCount, currentCannyPic)     # Count total Canny Edge for each pixel among all the image-stream

        pixelColorCount = np.add(
            pixelColorCount, getRGBColorPic(fileName))    # Count total RGB Color Sky Detection for each pixel among all the image-stream
        images_count += 1

    beta1 = 0.75
    beta2 = 0.5

    # Get rough sky region mask.
    getResultPic(resultPic, pixelColorCount, pixelEdgeCount, images_count * beta1, images_count * beta2)

    # cv2.imshow('result pic for sky region detection', resultPic)
    # cv2.waitKey(0)

    # Erode image to connect disjoint "non-sky region" 
    kernel = np.ones((5, 5), np.uint8)
    resultPic = cv2.erode(resultPic, kernel, iterations=1)

    # Find three largest external contours.
    (cnts, _) = cv2.findContours(
        resultPic.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:3]

    full_size = HEIGHT * WIDTH
    cnts_new = []

    index = 0
    for index in range(0, len(cnts)):
        if float(cv2.contourArea(cnts[index])) < float(full_size / 20):
            # Only consider contour region larger than 1/20 of full image size
            break
    cnts_new = cnts[:index]

    mask = np.zeros(resultPic.shape[:2], np.uint8)
    for mask_id in range(0, len(cnts_new)):
        cv2.drawContours(mask, cnts, mask_id, (mask_id + 1) * 80, -1)    # mark disjoint region with 80 gray interval

    contour_num = len(cnts_new)
    judge_result = [False] * contour_num

    # if we have more than one disjoint "sky region", we need to check all the other parts using CORRELATION COEFFICIENT. We assume the largest "sky region" is real and make it as base reference.
    if contour_num > 1:
        getCorrelationCoefficient(judge_result, start_pos, end_pos, step, images, mask)

        for x in range(0, HEIGHT):
            for y in range(0, WIDTH):
                grey_level = mask[x, y] / 80
                if grey_level == 1:
                    mask[x, y] = 255
                elif grey_level == 2 and judge_result[0] == False:
                    mask[x, y] = 0
                elif grey_level == 2 and judge_result[0] == True:
                    mask[x, y] = 255
                elif grey_level == 3 and judge_result[1] == False:
                    mask[x, y] = 0
                elif grey_level == 3 and judge_result[1] == True:
                    mask[x, y] = 255

    final = cv2.dilate(mask, kernel, iterations=1)

    mask2 = np.zeros(final.shape[:2], np.uint8)
    (cnts2, _) = cv2.findContours(
        final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts2 = sorted(cnts2, key=cv2.contourArea, reverse=True)[:contour_num]

    cv2.drawContours(mask2, cnts2, -1, 255, -1)

    # cv2.namedWindow('Sky Region')
    # cv2.imshow('sky region detection', mask2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return mask2


def getSunOrbit(dir_path1, dir_path2, dir_path3, sky_region_mask):
    """
    Get Sun Orbit among three-day image stream using SUN REGION MASK.

    Using dir_path2 image-stream as base directory. 
    1. Find matching image in dir_path1 and dir_path2. For example, if we have an image at 08-15-09-30-00 (mm-dd-hh-mm-ss) in dir_path, we need to find image in the range [08-14-09-28-00, 08-14-09-32-00] in dir_path1 and [08-16-09-28-00, 08-16-09-32-00] in dir_path3, namely 120 seconds time frame.
    2. Detect sun for each image-pair, for example (08-14-09-29-10, 08-15-09-30-00).
    3. Check sun intersection if both suns detected.
    4. Store image with intersected sun. These are images with real sun and could be used as generating sun orbit.
    """
    imglib_d1 = glob.glob(dir_path1 + '/*.jpg')
    imglib_d2 = glob.glob(dir_path2 + '/*.jpg')
    imglib_d3 = glob.glob(dir_path3 + '/*.jpg')

    imglib_d2_len = len(imglib_d2)

    pos_1 = 0
    pos_3 = 0

    start_pos_d2 = imglib_d2_len / 5
    end_pos_d2 = imglib_d2_len * 4 / 5

    count = 0

    sun_orbit_list = []

    for index_d2 in range(start_pos_d2, end_pos_d2):
        img_base_d2_filename = imglib_d2[index_d2]
        img_base_d2_filename_timeformat = convertDateTimeFormat(img_base_d2_filename)   # Convert file name to date time format

        list_1 = []
        list_3 = []

        # Find matching image in dir_path1
        list_1, pos_1 = getImageListInTimeFrame(imglib_d1, img_base_d2_filename_timeformat, start_pos=pos_1)
        # Find matching image in dir_path3
        list_3, pos_3 = getImageListInTimeFrame(imglib_d3, img_base_d2_filename_timeformat, start_pos=pos_3)

        flag_d1 = False
        flag_d3 = False

        # Detect sun intersection if sun detected
        if len(list_1) > 0 or len(list_3) > 0:
            img_base = sunDetect(img_base_d2_filename, sky_region_mask)
            if img_base is not None:
                if len(list_1) > 0:
                    img_d1 = sunDetect(list_1[0], sky_region_mask)
                    if img_d1 is not None and intersectionDetect(img_base, img_d1):
                        flag_d1 = True
                        sun_orbit_list.append(img_d1)
                if len(list_3) > 0:
                    img_d3 = sunDetect(list_3[0], sky_region_mask)
                    if img_d3 is not None and intersectionDetect(img_base, img_d3):
                        flag_d3 = True
                        sun_orbit_list.append(img_d3)
                if flag_d1 or flag_d3:
                    sun_orbit_list.append(img_base)
        if flag_d1 or flag_d3:
            count += 1
    return sun_orbit_list


def getCentroidList(sun_orbit_list):
    """
    Extract centroid of each sun.
    """
    centroid_list = []
    for i in sun_orbit_list:
        centroid_list.append(i[0])
    return centroid_list


matrix = None

def func(theta):
    global matrix
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    Rmat = np.reshape(np.array([cos_theta, -sin_theta, sin_theta, cos_theta]), (2, 2))  # Rotation Matrix
    centroid_rotate = np.dot(Rmat, matrix)
    coeffs, res, _, _, _ = np.polyfit(centroid_rotate[0,:], centroid_rotate[1,:], 2, full=True)
    return res


def getCoeff(theta, centroid):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    Rmat = np.reshape(np.array([cos_theta, -sin_theta, sin_theta, cos_theta]), (2, 2))
    centroid_rotate = np.dot(Rmat, centroid)
    coeffs, res, _, _, _ = np.polyfit(centroid_rotate[0,:], centroid_rotate[1,:], 2, full=True)
    return coeffs


def rotateClockWise(theta, x, y):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    Rmat = np.reshape(np.array([cos_theta, -sin_theta, sin_theta, cos_theta]), (2, 2))
    print 'In ClockWise Rotate--------------------'
    print 'Rmat = ', Rmat
    print 'Rmat_inv = ', np.linalg.inv(Rmat)
    plot_point = np.array(zip(x, y))
    plot_point = np.transpose(plot_point)
    print 'plot_point shape = ', plot_point.shape
    print 'plot_point = ', plot_point
    result = np.dot(np.linalg.inv(Rmat), plot_point)
    print 'result[0,:] = ', result[0, :]
    print 'result[1,:] = ', result[1, :]
    return result[0, :], result[1, :]


def rotateCounterClockWise(theta, x, y):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    Rmat = np.reshape(np.array([cos_theta, -sin_theta, sin_theta, cos_theta]), (2, 2))
    print 'In CounterClockWise Rotate--------------------'
    print 'Rmat = ', Rmat
    y = HEIGHT - y
    plot_point = np.reshape(np.array([x, y]), (2, 1))
    # plot_point = np.transpose(plot_point)
    print 'plot_point shape = ', plot_point.shape
    print 'plot_point = ', plot_point
    result = np.dot(Rmat, plot_point)
    print 'result[0,:] = ', result[0][0]
    print 'result[1,:] = ', result[1][0]
    return result[0][0], result[1][0]


def generalParabola(centroid):
    """
    Use fmin to minimize residual of polyfit function and return the rotation angular (theta)
    Reference :
    1. http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.optimize.fmin.html
    2. http://www.mathworks.com/matlabcentral/answers/80541-curve-fitting-general-parabola-translated-and-rotated-coordinate-system
    """
    global matrix
    x, y = zip(*centroid)
    y = tuple(HEIGHT - i for i in y)
    Centroid = zip(x, y)
    Centroid = np.transpose(np.asarray(Centroid))
    matrix = Centroid.copy()
    # theta = fmin(func, x0=0.79, args=(Centroid,))

    theta = fmin(func, x0=0.79)         # Get rotation angular

    coeffs = getCoeff(theta, Centroid)  # Get coefficients for parabola
    print 'optimized coefficient = ', coeffs
    polynomial = np.poly1d(coeffs)
    xs_1 = np.arange(-10000, 10000, 0.5)
    ys_1 = polynomial(xs_1)
    xs_2, ys_2 = rotateClockWise(theta, xs_1, ys_1)
    plot(x, y, 'o')
    # plot(xs_1, ys_1, '-g')
    plot(xs_2, ys_2, '.r')
    axis([0, WIDTH, 0, HEIGHT])
    ylabel('y')
    xlabel('x')
    title('General Paraloba Fitting')
    show()
    return theta, coeffs


def convertDateTimeFormat(file_name):
    """
    Convert file name to date time format.
    e.g. 04E6768320C1_07-31-2015_10-19-09 to timedelta format
    """
    date_time = timedelta(hours=int(
        file_name[-12:-10]), minutes=int(file_name[-9:-7]), seconds=int(file_name[-6:-4]))
    return date_time


def getImageListInTimeFrame(imglib, base_time, time_frame=120, start_pos=0):
    """
    Find matching image within time_frame
    """
    img_list = []

    if start_pos >= len(imglib):
        return img_list

    if start_pos > 0:
        start_pos -= 1

    for index in range(start_pos, len(imglib)):
        img_current = imglib[index]
        current_time = convertDateTimeFormat(img_current)
        if current_time < base_time and (base_time - current_time).total_seconds() > time_frame:
            continue
        elif current_time >= base_time and (current_time - base_time).total_seconds() > time_frame:
            start_pos = index
            break
        else:
            img_list.append(img_current)
            start_pos = index

    return img_list, start_pos


def sunDetect(file_name, mask):
    """
    Detect Sun of an image using hue channel under sky region mask.
    Generally, sun is regared as extremely white (different from cloud white) in rgb space. We observed that the hue value of sun region usually less than 10. So we use hue channel to distinguish sun region and cloud region. 
    Future To Do: could use combination of Hue channel and Value channel in HSV space.
    After we detect "Sun Region" using hue channel, we use minEnclosingCircle in OpenCV lib to enclose the region with minimum circle. If the "Sun Region" is larger than half of the minEnclosingCircle area AND the circle radius is in range [15px, 100px], we keep the largest region as Sun!
    """
    img_bgr = cv2.imread(file_name)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    h_channel = img_hsv[:, :, 0]
    # cv2.imshow('h channel', h_channel)
    h_channel[h_channel <= 10] = 0
    h_channel[h_channel > 10] = 255
    h_channel = (255 - h_channel)

    h_channel = cv2.bitwise_and(mask, h_channel)

    kernel = np.ones((3, 3), np.uint8)
    h_channel = cv2.erode(h_channel, kernel, iterations=1)
    # cv2.imshow('h_channel_after_erode', h_channel)

    mask_h = np.zeros(h_channel.shape[:2], np.uint8)

    (cnts_h, _) = cv2.findContours(
        h_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_h = sorted(cnts_h, key=cv2.contourArea, reverse=True)[:1]

    cv2.drawContours(mask_h, cnts_h, -1, 255, -1)

    mask_h = cv2.dilate(mask_h, kernel, iterations=1)
    # cv2.imshow('h_channel_after_dilate', mask_h)

    (cnts_h, _) = cv2.findContours(
        mask_h, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_h = sorted(cnts_h, key=cv2.contourArea, reverse=True)[:1]

    cv2.drawContours(mask_h, cnts_h, -1, 255, -1)

    mask_h = cv2.medianBlur(mask_h, 15)

    # cv2.drawContours(img_bgr, cnts_h, -1, 255, 2)
    # cv2.imshow('mask', img_bgr)
    # cv2.waitKey(0)

    if len(cnts_h) > 0:
        (x_h, y_h), radius_h = cv2.minEnclosingCircle(cnts_h[0])
        center_h = (int(x_h), int(y_h))
        radius_h = int(radius_h)

        mask_test_h = np.zeros((HEIGHT, WIDTH), np.uint8)
        cv2.circle(mask_test_h, center_h, radius_h, 255, -1)
        mask_test_h = cv2.bitwise_and(mask, mask_test_h)
        (cnts_test_h, _) = cv2.findContours(
            mask_test_h.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_test_h = sorted(cnts_test_h, key=cv2.contourArea, reverse=True)[:1]
        cv2.drawContours(img_bgr, cnts_test_h, -1, 255, 3)

        contour_area = cv2.contourArea(cnts_h[0])
        enclosing_area = cv2.contourArea(cnts_test_h[0])
        percentage = contour_area / enclosing_area

        #
        if radius_h >= 15 and radius_h <= 100 and percentage >= 0.5:
            # cv2.imshow('Sun Detection', img_bgr)
            # cv2.waitKey(0)
            return (center_h, radius_h)


def intersectionDetect(t1, t2):
    """
    Detect sun intersection.
    """
    mask_1 = np.zeros((HEIGHT, WIDTH), np.uint8)
    mask_2 = np.zeros((HEIGHT, WIDTH), np.uint8)

    cv2.circle(mask_1, t1[0], t1[1], 255, -1)
    cv2.circle(mask_2, t2[0], t2[1], 255, -1)

    intersection = cv2.bitwise_and(mask_1, mask_2)
    (cnt, _) = cv2.findContours(intersection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(cnt, key=cv2.contourArea, reverse=True)[:1]
    if len(cnt) > 0:
        intersection_area = cv2.contourArea(cnt[0])
        full_size = HEIGHT*WIDTH
        percentage = intersection_area / full_size
        if percentage > 0:
            return True
        else:
            return False

    return False


coefficient_equations = None

def equations(p):
    print 'coeff = ', coefficient_equations
    x, y = p
    cx = coefficient_equations[0][0][0]
    cy = coefficient_equations[0][0][1]
    r = coefficient_equations[0][1]
    a = coefficient_equations[1][0]
    b = coefficient_equations[1][1]
    c = coefficient_equations[1][2]
    print 'cx = ', cx, 'cy = ', cy, 'r = ', r, 'a = ', a, 'b = ', b, 'c = ', c
    return ((x-cx)*(x-cx)+(y-cy)*(y-cy)-r*r, a*x*x+b*x+c-y)


def solveEquations(p1, p2):
    '''
    p1: circle equation parameter: (x-cx)^2 + (y-cy)^2 = r^2 -----((cx,cy),r)
    p2: quadratic equation parameter: y = a*x^2+b*x+c ----- (a,b,c)
    '''
    global coefficient_equations
    coefficient_equations = (p1, p2)
    x, _, ier, mesg = fsolve(equations, (p1[0][0], p1[0][1]), full_output=1)
    print 'ier = ', ier
    print 'mesg =', mesg
    return ier


def drawResult(p1, p2):
    circle = Circle((p1[0][0], p1[0][1]), p1[1], color='b', fill=False)
    polynomial = np.poly1d(p2)
    xs_1 = np.arange(-10000, 10000, 0.5)
    ys_1 = polynomial(xs_1)
    plot(xs_1, ys_1, '-g')
    axis([-1000, 1000, -1000, 1000])
    ylabel('y')
    xlabel('x')
    title('General Paraloba Fitting')
    fig = gcf()
    fig.gca().add_artist(circle)
    show()


def sunDetectUsingOrbit(img, theta, coeffs):
    """
    Check if the sun in the image is a real one:
    1. Rotate the centroid of the sun to the adjusted coordinate plane.
    2. Solve the equation of circle and quadratic to check solution.
    """
    center = img[0]
    print 'center = ', center
    x, y = rotateCounterClockWise(theta, center[0], center[1])
    new_center = (x, y)
    print 'new_center = ', new_center
    p1 = (new_center, img[1])
    p2 = coeffs
    isIntersect = solveEquations(p1, p2)
    drawResult(p1, p2)
    return True if isIntersect == 1 else False


def calculateCloudCoverage(filename, sky_region_mask, sun=None):
    print 'sun = ', sun
    img = cv2.imread(filename)
    sky_area = 0
    cloud_area = 0
    for x in range(0, HEIGHT):
        for y in range(0, WIDTH):
            if sky_region_mask[x, y] == 255:
                sky_area += 1
                if criteria(img[x, y]) == True:
                    sky_region_mask[x, y] = 0
                    cloud_area += 1
    if sun is not None:
        center = sun[0]
        radius = sun[1]
        cloud_area -= (np.pi*radius*radius)
        cv2.circle(sky_region_mask, center, radius, 255, -1)
    return float(cloud_area)/float(sky_area)


def criteria(pixel):
    b = np.int16(pixel[0])
    g = np.int16(pixel[1])
    r = np.int16(pixel[2])
    if np.absolute(r - b) < 30 and np.absolute(b - g) < 30:
        return True
    else:
        return False
