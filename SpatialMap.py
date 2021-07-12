import numpy as np
from enum import IntEnum
import rospy
from nav_msgs.msg import OccupancyGrid

class GridMap:     
    def __init__(self, width, height):
        self.rate = rospy.Rate(1)
        self.map = OccupancyGrid();
        self.map.header.stamp = rospy.Time.now()
        self.map.header.frame_id = "map"      
        self.map.info.resolution = 1
        self.map.info.width = width
        self.map.info.height = height
        self.map.info.origin.position.x = 0
        self.map.info.origin.position.y = 0
        self.map.info.origin.position.z = 0
        self.map.info.origin.orientation.x = 0
        self.map.info.origin.orientation.y = 0
        self.map.info.origin.orientation.z = 0
        self.map.info.origin.orientation.w = 1
        
        
    def xy_to_index(self, x, y):
        index = x + self.map.info.width * y
        return index
    
    def index_to_xy(self, index):
        x = index %  self.map.info.width
        y = index // self.map.info.width
        return x, y
    
    def update_cell(self, pos, value):
        #print(pos)
        i = self.xy_to_index(pos[0], pos[1])
        self.map.data[i] = value
        
    def ros_publish_map(self):
        self.spatial_map.header.stamp = rospy.Time.now()
        self.map_pub.publish(self.spatial_map)




class SpatialMap(GridMap):
    class OccGridStates(IntEnum):
        free = 0
        occupied = 100
        unknown = -1

    def __init__(self, width, height):
        super().__init__(width, height)
        self.factor = 3
        self.map.info.resolution = self.factor
        self.map.data = np.full(width*height, self.OccGridStates.unknown)

class ObjectMap(GridMap):
    class ObjGridStates(IntEnum):
        box = 110
        ball = -2
        key = 80
        checked_box = -120
        floor = 0
        unknown = -1
        wall = 100
        goal = 50
        """
        wall = 120
        box = 100
        ball = 80
        key = 60
        floor = 0
        unknown = -1
        """
    def __init__(self, width, height, factor):
        self.factor = factor # must be odd
        super().__init__(width * self.factor, height * self.factor)
        self.map.data = np.full(width*height*self.factor*self.factor, self.ObjGridStates.unknown)

    def xy_to_index(self, x, y):
        index = x + self.map.info.width // self.factor * y
        return index

    def index_to_xy(self, index):
        x = index %  (self.map.info.width // self.factor)
        y = index // (self.map.info.width // self.factor)
        return x, y

    def update_cell(self, pos, value, center_only = False):
        #print(pos)
        i = self.xy_to_index(pos[0], pos[1])
        self.update_surrounding(i, value, center_only)
        
    def update_surrounding(self, index, value, center_only = False):
        center_index = (index // (self.map.info.width//self.factor)) * self.map.info.width*self.factor + (self.factor//2) * self.map.info.width + (index % (self.map.info.width//self.factor)) * self.factor + (self.factor//2)
        #print(index)
        #print(center_index)
        if not center_only:
            self.update_upleft(center_index, value)
            self.update_upmid(center_index, value)
            self.update_upright(center_index, value)
            self.update_left(center_index, value)
            self.update_right(center_index, value)
            self.update_downleft(center_index, value)
            self.update_downmid(center_index, value)
            self.update_downright(center_index, value)
  
        elif self.map.data[center_index-1] != self.ObjGridStates.box:
            self.update_upleft(center_index, self.ObjGridStates.floor)
            self.update_upmid(center_index, self.ObjGridStates.floor)
            self.update_upright(center_index, self.ObjGridStates.floor)
            self.update_left(center_index, self.ObjGridStates.floor)
            self.update_right(center_index, self.ObjGridStates.floor)
            self.update_downleft(center_index, self.ObjGridStates.floor)
            self.update_downmid(center_index, self.ObjGridStates.floor)
            self.update_downright(center_index, self.ObjGridStates.floor)
       
        self.update_center(center_index, value)

    def update_upleft(self, center_index, value):
        upleft_index =  center_index - self.map.info.width - 1
        self.map.data[upleft_index] = value
        
    def update_upmid(self, center_index, value):
        upmid_index =  center_index - self.map.info.width
        self.map.data[upmid_index] = value
        
    def update_upright(self, center_index, value):
        upright_index =  center_index - self.map.info.width + 1
        self.map.data[upright_index] = value
        
    def update_left(self, center_index, value):
        left_index =  center_index - 1
        self.map.data[left_index] = value
        
    def update_right(self, center_index, value):
        right_index =  center_index + 1
        self.map.data[right_index] = value

    def update_downleft(self, center_index, value):
        downleft_index = center_index + self.map.info.width - 1
        self.map.data[downleft_index] = value
        
    def update_downmid(self, center_index, value):
        downmid_index = center_index + self.map.info.width
        self.map.data[downmid_index] = value
        
    def update_downright(self, center_index, value):
        downright_index = center_index + self.map.info.width + 1
        self.map.data[downright_index] = value
        
    def update_center(self, center_index, value):
        self.map.data[center_index] = value
        
class BinaryMap(ObjectMap):
    def __init__(self, width, height, value = 0):
        self.factor = 5 # must be odd
        super().__init__(width, height, self.factor)
        self.map.info.resolution = 1
        self.map.data = np.full(width*height*self.factor*self.factor, value)
        #print(self.map.info.width)
        
    def update_surrounding(self, index, value, center_only = False):
        center_index = (index // (self.map.info.width//self.factor)) * self.map.info.width*self.factor + (self.factor//2) * self.map.info.width + (index % (self.map.info.width//self.factor)) * self.factor + (self.factor//2)
        #print("index: {}, center_index: {}".format(index, center_index))
        near_5by5 = []
        for i in range(25):
            near_5by5.append(center_index + self.map.info.width*(i//5-2) + (i%5-2))
            
        for j in range(25):
            self.map.data[near_5by5[j]] = value
