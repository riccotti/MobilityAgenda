import json
import datetime

from util.mobility_distance_functions import *


__author__ = 'Riccardo Guidotti'


class Trajectory:
    """description"""

    def __init__(self, id, object, vehicle, length=None, duration=None):
        self.id = id
        self.object = object
        self.vehicle = vehicle
        self._length = length
        self._duration = duration

    def id(self):
        return self.id

    def object(self):
        return self.object

    def vehicle(self):
        return self.vehicle

    def num_points(self):
        return self.__len__()

    def point_n(self, n):
        return self.object[n]

    def start_point(self):
        return self.point_n(0)

    def end_point(self):
        return self.point_n(len(self)-1)

    def length(self):
        if self._length is None:
            length = 0
            for i in range(0, len(self.object)-1, 1):
                p1 = self.point_n(i)
                p2 = self.point_n(i+1)
                dist = spherical_distance(p1, p2)
                length += dist
            self._length = length
        return self._length

    def duration(self):
        if self._duration is None:
            duration = 0
            for i in range(0, len(self.object) - 1, 1):
                p1 = self.point_n(i)
                p2 = self.point_n(i + 1)
                dist = p2[2] - p1[2]
                duration += dist
            self._duration = duration
        return self._duration

    def __len__(self):
        return len(self.object)

    def __str__(self):
        return json.dumps({'id': self.id, 'vehicle': self.vehicle, 'object': self.object})

    def __repr__(self):
        return self.__str__()

    def __unicode__(self):
        return self.__str__()



