from __future__ import print_function

import math
# from sensor_msgs.msg import PointCloud2
# import roslib.message
# import numpy as np


M_PI = 3.14159265358979323846
lowerBound = -24.9


def creatNanList(num):
    nanList = []
    for i in range(num):
        nanList.append([])
    return nanList

# def read_points(cloud, field_names=None, skip_nans=False, uvs=[]):
#     fmt = '<fff'
#     width, data = cloud.width, cloud.data
#     unpack_from = struct.Struct(fmt).unpack_from

#     pointcloud = []
#     scanIDs = []

#     offset = 0
#     for u in range(width):
#         p = unpack_from(data, offset)

#         if (p[1] < 7) & (p[1] > -10) & (p[0] < -7) & (p[0] > -25) & (p[2] < -1.5):
#             scanID = hiPoints_WhereAreYouFrom(p)
#             # if ((scanID >= 64) | (scanID < 0)):
#             #     continue
#             if scanID == 44:
#                 pointcloud.append(p)
#                 scanIDs.append(scanID)
#         offset += 16

#     return pointcloud, scanIDs


def hiPoints_WhereAreYouFrom(pt):
    angle = math.atan(pt[2] / math.sqrt(pt[0] * pt[0] + pt[1] * pt[1]))
    # scanID = int(((angle * 57.29577951308232) + 24.9) * 2.342007434944238 + 0.5)
    scanID = int(angle * 134.18714161056457 + 58.81598513011153)
    return scanID


def giveYou_MaxAndMin(aList):
    aList_max = aList_min = aList[0][2]

    for l in aList:
        if l[2] < aList_min:
            aList_min = l[2]
        if l[2] > aList_min:
            aList_max = l[2]

    return aList_max, aList_min


def getTheFarDistance(aList):
    distance = 0
    idx = 0
    lenOfList = len(aList)
    
    for i in range(lenOfList-1):
        dis = abs(aList[i][2] - aList[i+1][2])
        if dis > distance:
            distance = dis
            idx = i

    return distance, idx
