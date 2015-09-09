import cv2
from Tkinter import Tk
from tkFileDialog import askopenfilename
from tkFileDialog import askdirectory
import ImageProcess


HEIGHT = 640
WIDTH = 640


if __name__ == "__main__":
    """
    1. Use image-stream from one specific day to generate SKY REGION MASK.
    2. Use image-stream from three days to generate SUN ORBIT under SKY REGION MASK.
    3. Check "Real Sun" on the test image with help of SUN ORBIT
    4. Calculate Cloud Cover
    """
    
    # Read Three-day image directory from local
    root = Tk()
    root.withdraw()
    filename = askopenfilename()	# Get test image path
    print filename
    path_1 = askdirectory()			# Get First Day Image
    path_2 = askdirectory()			# Get Second Day Image
    path_3 = askdirectory()			# Get Third Day Image
    root.destroy()
    path = filename.rsplit('/', 1)[0]
    print path

    # Get SKY REGION MASK
    sky_region_mask = ImageProcess.getSkyRegion(path)

    # Get Sun Orbit among three-day image stream using SUN REGION MASK
    sun_orbit = []
    sun_orbit = ImageProcess.getSunOrbit(path_1, path_2, path_3, sky_region_mask)

    # Get Centroid of all the Detected Sun in sun_orbit.
    centroid_list = []
    centroid_list = ImageProcess.getCentroidList(sun_orbit)

    # Fit the centroid list sample to a quadratic equation(general parabola).
    theta, coeffs = ImageProcess.generalParabola(centroid_list)

    # Detect sun of test image
    img_centroid_radius = ImageProcess.sunDetect(filename, sky_region_mask)

    # percentage = 0.0

    if img_centroid_radius is not None and ImageProcess.sunDetectUsingOrbit(img_centroid_radius, theta, coeffs):
        # Sun Detected!
        percentage = ImageProcess.calculateCloudCoverage(filename, sky_region_mask, img_centroid_radius)
    else:
        percentage = ImageProcess.calculateCloudCoverage(filename, sky_region_mask)

    print 'Cloud Coverage = ', percentage
    cv2.imshow('final result', sky_region_mask)
    cv2.waitKey(0)
