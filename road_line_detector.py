import rospy
import numpy as np
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge, CvBridgeError
import cv2
import time
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg
import pickle


x_pixels_per_meter = 220 / 7.0
y_pixels_per_meter = 380 / 5.3

pkl_file = open('data.pkl', 'rb')
fields = pickle.load(pkl_file)
pkl_file.close()


class lineFinder():
    def __init__(self, isLeftLine_=True):
        self.first = True
        self.found = False
        self.coeffs = np.array([], dtype=np.float64)
        self.coeffs_final = []
        self.initial_coeffs = np.array([], dtype=np.float64)
        self.next_coeffs = np.array([], dtype=np.float64)
        self.isLeftLine = isLeftLine_
        self.nextMargin = 10
        self.points = []

        self.leastedStableLeftLines_5 = []

        # self.coor_offset_f = 16.45
        # self.coor_offset_l = 3.65 + 1.42

    def find_lane_line(self, mask):
        if self.first:
            self.get_initial_coeffs(mask, self.isLeftLine)
            if self.initial_coeffs is None:
                return [], []
            self.next_coeffs = self.initial_coeffs
            self.coeffs = self.next_coeffs
            self.first = False
            self.leastedStableLeftLines_5.append(self.coeffs)
        else:
            self.get_next_coeffs(mask, self.next_coeffs)

            if (abs(self.leastedStableLeftLines_5[-1][0] - self.next_coeffs[0]) < 0.08) & (abs(self.leastedStableLeftLines_5[-1][1] - self.next_coeffs[1]) < 0.15):
                self.coeffs = self.next_coeffs
                if len(self.leastedStableLeftLines_5) == 5:
                    self.leastedStableLeftLines_5 = self.leastedStableLeftLines_5[1:5]
                self.leastedStableLeftLines_5.append(self.coeffs)
            else:
                self.coeffs = self.leastedStableLeftLines_5[-1]
                self.first = True

        self.changeToLidarCoord()
        self.returnPointCloudLine()
        point_out = self.points
        self.points = []

        return point_out, self.coeffs

    def get_next_coeffs(self, mask, coeffs):
        nonzero = mask.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        lane_inds = ((nonzerox > coeffs[0] * nonzeroy + coeffs[1] - self.nextMargin)) & \
                    (nonzerox < coeffs[0] * nonzeroy + coeffs[1] + self.nextMargin)

        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        self.next_coeffs = np.polyfit(y, x, 1)

    def get_initial_coeffs(self, mask, kind):
        histogram = np.sum(mask[150:, :], axis=0)
        if kind:
            x_base = np.argmax(histogram[:240])
        else:
            x_base = np.argmax(histogram[240:]) + 240

        nonzero = mask.nonzero()
        nonzeroy = nonzero[0]
        nonzerox = nonzero[1]

        x_current = x_base
        n_windows = 9
        window_height = np.int(300 // n_windows)
        minpix = 50
        lane_inds = []

        for level in range(n_windows):
            win_y_low = 300 - (level + 1) * window_height
            win_y_high = 300 - (level * window_height)
            win_x_low = x_current - 30
            win_x_high = x_current + 30

            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]

            if len(good_inds) > minpix:
                x_current = np.int(np.mean(nonzerox[good_inds]))

            if len(good_inds) > 5:
                lane_inds.append(good_inds)

        try:
            lane_inds = np.concatenate(lane_inds)
            x = nonzerox[lane_inds]
            y = nonzeroy[lane_inds]
            self.initial_coeffs = np.polyfit(y, x, 1)
        except:
            self.initial_coeffs = None

    def get_coeffs(self):
        return self.coeffs

    def changeToLidarCoord(self):
        # realWorldDistCoeffs
        a_0 = self.coeffs[0] * x_pixels_per_meter / y_pixels_per_meter
        b_0 = self.coeffs[1] / y_pixels_per_meter
        # move_to_zuoshang_zhuitong
        # a_1 = a_0
        b_1 = b_0 - 0.6973684211
        # change_to_lidar_coord
        # fenmu = 0.99949 - (0.031903 * a_1)
        # a_2 = (0.99949 * a_1 + 0.031903) / fenmu
        # b_2 = (16.45 * a_1 + b_1 -3.65) / fenmu
        b_2 = - (11.5 * a_0 + b_1 - 3.3)
        # a_ = realWorldDistCoeffs[0]
        # b_ = realWorldDistCoeffs[0] * self.coor_offset_f + realWorldDistCoeffs[1] - self.coor_offset_l

        self.coeffs_final = [a_0, b_2]

    def returnPointCloudLine(self):
        for i in range(120):
            x = i / 10
            y = self.coeffs_final[0] * x + self.coeffs_final[1]
            z = -1.5

            point = (x, y, z)
            self.points.append(point)


class start_gettingImageAndDetecteRoadLine():
    def __init__(self):
        self.bridge = CvBridge()
        self.laneDetector = laneDetector()
        rospy.init_node('roadline_detector', anonymous=True)
        self.pub_ = rospy.Publisher('roadlineDetectionResults', PointCloud2, queue_size=10)
        self.sub_ = rospy.Subscriber('/pylon_camera_node/image_raw', Image, self.callback_image)
        rospy.spin()

    def callback_image(self, data):
        pc_header = std_msgs.msg.Header()
        pc_header.frame_id = 'velodyne'
        pc_header.stamp = rospy.Time.now()
        start = time.time()
        cv_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        cv_img = cv2.resize(cv_img, (480, 300), interpolation=cv2.INTER_AREA)
        # detected_points, detected_image = self.laneDetector.detector(cv_img)
        detected_points, detected_image, mask_ = self.laneDetector.detector(cv_img)

        data_output = pc2.create_cloud(pc_header, fields, points=detected_points)
        # self.pub_.publish(data_output)

        cv2.imshow("frame", detected_image)
        cv2.imshow("frame2", mask_)
        cv2.waitKey(3)
        end = time.time()
        FPS = 1 / (end - start)
        print(FPS)


class laneDetector():
    def __init__(self):
        self.raw_image = None

        self.small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 4))
        self.small_kernel_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 4))
        self.large_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 12))
        # self.src_points = np.array([[30., 95.], [321., 95.], [112., 50.], [268., 50.]], dtype="float32")
        # self.dst_points = np.array([[90., 300.], [390., 300.], [90., 0.], [390., 0.]], dtype="float32")
        self.src_points = np.array([[66., 166.], [359., 163.], [164., 111.], [287., 107.]], dtype="float32")
        self.dst_points = np.array([[50., 220.], [430., 220.], [50., 0.], [430., 0.]], dtype="float32")
        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)

        self.laneWidth = 4.74
        self.mask = None

        self.leftLine = lineFinder(True)
        self.rightLine = lineFinder(False)

    def calculateMask(self):
        image = cv2.warpPerspective(self.raw_image, self.M, (480, 300), cv2.INTER_LINEAR)
        image = cv2.medianBlur(image, 5)

        image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        # sobelx = np.abs(cv2.Sobel(image_lab[:, :, 2], cv2.CV_64F, 1, 0, ksize=5))
        # roi_edge = cv2.inRange(sobelx, (125), (255))
        # roi_edge = cv2.morphologyEx(roi_edge, cv2.MORPH_ERODE, self.small_kernel_1)
        # roi_edge_2 = cv2.morphologyEx(roi_edge, cv2.MORPH_DILATE, self.large_kernel)
        roi_mask_lab = cv2.inRange(image_lab[:, :, 2], (0), (125))
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        roi_mask1 = cv2.inRange(image_hsv, (3, 0, 0), (21, 64, 255))
        roi_mask2 = cv2.inRange(image_hsv, (150, 0, 0), (179, 64, 255))
        roi_mask_hsv = roi_mask1 | roi_mask2

        # mask = roi_mask & roi_edge_2 | roi_edge
        self.mask = roi_mask_lab | roi_mask_hsv
        # self.mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, self.small_kernel)

    def detector(self, raw_image_):
        self.raw_image = raw_image_
        self.calculateMask()

        self.leftLine.find_lane_line(self.mask)
        detection_line_points_left, coeffs_left = self.leftLine.find_lane_line(self.mask)

        self.rightLine.find_lane_line(self.mask)
        detection_line_points_right, coeffs_right = self.rightLine.find_lane_line(self.mask)

        fitx_right, ploty_right = self.get_line_pts(coeffs_right)
        fitx_left, ploty_left = self.get_line_pts(coeffs_left)

        line_left = self.draw_lines(self.mask, fitx_left, ploty_left)
        line_right = self.draw_lines(self.mask, fitx_right, ploty_right)
        lineImage = line_left + line_right
        lineMask = cv2.warpPerspective(lineImage, self.M, (480, 300), flags=cv2.WARP_FILL_OUTLIERS + cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP)
        imageInit_1 = self.raw_image * (1-lineMask)
        imageInit_1[..., 1] = imageInit_1[..., 1] + (lineMask[..., 1] * 40)
        return detection_line_points_left + detection_line_points_right, imageInit_1, self.mask

        # return detection_line_points_left + detection_line_points_right, self.mask

    def get_line_pts(self, coeffs):
        # 算出线的坐标值
        ploty = np.linspace(0, 300 - 1, 300)
        fitx = coeffs[0] * ploty + coeffs[1]
        return fitx, ploty

    def draw_pw(self, img, pts, color):
        # 在图上根据坐标画线
        pts = np.int_(pts)
        for i in range(len(pts) - 1):
            x1 = pts[i][0]
            y1 = pts[i][1]
            x2 = pts[i+1][0]
            y2 = pts[i+1][1]
            cv2.line(img, (x1, y1), (x2, y2), color, 9)
        return img

    def draw_lines(self, mask, fitx, ploty):
        color = (1, 1, 1)

        out_img = np.dstack((mask, mask, mask))
        window_img = np.zeros_like(out_img)

        lane_points = np.array(list(zip(fitx, ploty)), np.int32)
        out_img = self.draw_pw(window_img, lane_points, color)
        return out_img


if __name__ == '__main__':
    start_gettingImageAndDetecteRoadLine()
