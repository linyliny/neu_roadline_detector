import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import time
import util
import struct
import numpy as np
import pickle

# 传输的点直接把区域画好，减少点的数量，提高效率。


class subscribeAndPublish():
    # '''
    # 这一部分主要负责从ros话题中读取pointcloud2格式的点云数据，
    # 并且按照区域、环数，左右进行取舍和区分，存到dipartCpsLeft、dipartCpsRight中。
    # 其中一级list为不同环数，list中的list为存储的各个点。
    # 然后用这些此数据去执行curb检测。
    # '''
    def __init__(self):
        rospy.init_node('curbDetection', anonymous=True)

        self.unpack_from = struct.Struct('<fff').unpack_from
        self.pub_ = rospy.Publisher('curbDetectionResults', PointCloud2, queue_size=10)
        # self.sub_ = rospy.Subscriber('/velodyne_points', PointCloud2, self.callback_pointcloud)
        self.sub_ = rospy.Subscriber('/neuavslam/groudPointCloud', PointCloud2, self.callback_pointcloud)

        self.x_range_up = -14  # -14
        self.x_range_down = -7  # -9
        self.ring_up = 40  # 42
        self.ring_down = 33  # 35
        self.ring_up_to_down = self.ring_up - self.ring_down

        self.dipartCpsLeft = util.creatNanList(self.ring_up_to_down)
        self.dipartCpsRight = util.creatNanList(self.ring_up_to_down)

    def callback_pointcloud(self, data):
        header = data.header
        fields = data.fields
        output = open('data.pkl', 'wb')
        pickle.dump(fields, output, protocol=2)
        output.close()
        start = time.time()
        self.read_points(data)

        curb_detector = curbDetector(self.dipartCpsRight, self.dipartCpsLeft)
        curb_detector.getCurbPoints()
        curb_detector.curbDetectionResultsVisualize()

        data_output = pc2.create_cloud(header, fields, curb_detector.curbPointsLeft + curb_detector.curbPointsRight)
        # data_output = pc2.create_cloud(header, fields, curb_detector.IMissYou)

        self.pub_.publish(data_output)

        # print(curb_detector.lineRight)

        end = time.time()
        print(end-start)

        self.dipartCpsLeft = util.creatNanList(self.ring_up_to_down)
        self.dipartCpsRight = util.creatNanList(self.ring_up_to_down)

    def read_points(self, cloud):
        width, data = cloud.width, cloud.data
        last_point = 0
        # downSampling = 0

        offset = 0
        for u in range(width):
            # if (downSampling % 3) == 0:
            #     downSampling += 1
            # else:
            #     offset += 16
            #     downSampling += 1
            #     continue

            p = self.unpack_from(data, offset)

            if last_point == p:
                offset += 16
                continue

            if (p[1] < 7) & (p[1] > -10) & (p[0] < self.x_range_down) & (p[0] > self.x_range_up) & (p[2] < -1.5):
                scanID = util.hiPoints_WhereAreYouFrom(p)
                if ((scanID >= self.ring_up) | (scanID < self.ring_down)):
                    offset += 16
                    continue
                if p[1] > 0:
                    self.dipartCpsRight[scanID-self.ring_down].append(p)
                else:
                    self.dipartCpsLeft[scanID-self.ring_down].append(p)

            last_point = p

            offset += 16


class curbDetector():
    def __init__(self, dipartCpsRight, dipartCpsLeft):
        self.dipartCpsRight = dipartCpsRight
        self.dipartCpsLeft = dipartCpsLeft
        self.dipartCpsNum = len(self.dipartCpsRight)
        self.curbPointsLeft = []
        self.curbPointsRight = []

        self.lineRight = None
        self.lineLeft = None

        self.curbDetectionResultsVisualizePoints = []

        self.IMissYou = []

    def getCurbPoints(self):
        for lineIdx in range(self.dipartCpsNum):
            self.getCurbPoint(lineIdx, 'Right')
            self.getCurbPoint(lineIdx, 'Left')

        self.theLine()

    def getCurbPoint(self, idx, which_line):
        if which_line == 'Right':
            sortedPoint = sorted(self.dipartCpsRight[idx], key=lambda x:x[1])

        if which_line == 'Left':
            sortedPoint = sorted(self.dipartCpsLeft[idx], key=lambda x:x[1], reverse=True)

        cleanedPoints = self.piao_rou(sortedPoint)
        self.IMissYou = self.IMissYou + sortedPoint
        self.slideForGettingPoints(cleanedPoints, which_line)

    def slideForGettingPoints(self, points, whichLine):
        w_0 = 10
        w_d = 30
        i = 0

        # some important parameters influence the final performance.
        xy_thresh = 0.1  # 0.1
        z_thresh = 0.06  # 0.05

        points_num = len(points)

        while((i + w_d) < points_num):
            w_max, w_min = util.giveYou_MaxAndMin(points[i:(i + w_d)])
            if abs(w_max - w_min) >= z_thresh:
                for ii in range(w_d-1):
                    p_dist = (((points[i + ii][1] - points[i + 1 + ii][1]) **2) + ((points[i + ii][0] - points[i + 1 + ii][0]) **2)) ** 0.5
                    if p_dist >= xy_thresh:
                        if whichLine == 'Right':
                            self.curbPointsRight.append(points[ii + i])
                            return 0
                        else:
                            self.curbPointsLeft.append(points[ii + i])
                            return 0

                _, max_idx = util.getTheFarDistance(points[i:(i + w_d)])
                if whichLine == 'Right':
                    self.curbPointsRight.append(points[max_idx + i])
                    return 0
                else:
                    self.curbPointsLeft.append(points[max_idx + i])
                    return 0

            i += w_0

    def theLine(self):
        right_num = len(self.curbPointsRight)
        left_num = len(self.curbPointsLeft)

        if right_num > 1:
            right_x = np.zeros(right_num)
            right_y = np.zeros(right_num)
            for rr in range(right_num):
                right_x[rr] = self.curbPointsRight[rr][0]
                right_y[rr] = self.curbPointsRight[rr][1]
            self.lineRight = np.polyfit(right_x, right_y, 1)

        if left_num > 1:
            left_x = np.zeros(left_num)
            left_y = np.zeros(left_num)
            for ll in range(left_num):
                left_x[ll] = self.curbPointsLeft[ll][0]
                left_y[ll] = self.curbPointsLeft[ll][1]
            self.lineLeft = np.polyfit(left_x, left_y, 1)

    def curbDetectionResultsVisualize(self):
        if self.lineRight is not None:
            rx = np.arange(-25, -7, 0.1)
            ry = self.lineRight[0] * rx + self.lineRight[1]

            for i in range(len(rx)):
                point_new = (rx[i], ry[i], 0)
                self.curbDetectionResultsVisualizePoints.append(point_new)

        if self.lineLeft is not None:
            lx = np.arange(-25, -7, 0.1)
            ly = self.lineLeft[0] * lx + self.lineLeft[1]

            for i in range(len(lx)):
                point_new = (lx[i], ly[i], 0)
                self.curbDetectionResultsVisualizePoints.append(point_new)

    def piao_rou(self, points):
        jump_flag = False
        cleanedHair = []
        mayDropedHair = []
        numOfTheHair = len(points)

        xy_treth = 0.3
        z_treth = 0.08
        N = 20

        for j in range(numOfTheHair-1):
            i = j + 1
            p_dist = (((points[i][0] - points[i-1][0]) ** 2) + ((points[i][1] - points[i-1][1]) ** 2)) ** 0.5
            z_high = abs(points[i][2] - points[i-1][2])

            if (p_dist <= xy_treth) & (z_high <= z_treth):
                if jump_flag:
                    mayDropedHair.append(points[i])
                    if (len(mayDropedHair) >= N):
                        cleanedHair = cleanedHair + mayDropedHair
                        jump_flag = False
                        mayDropedHair = []
                else:
                    cleanedHair.append(points[i])
            else:
                jump_flag = True
                mayDropedHair = []
                mayDropedHair.append(points[i])

        return cleanedHair


if __name__ == '__main__':
    curb_detection = subscribeAndPublish()
    # print('hi')
    rospy.spin()
