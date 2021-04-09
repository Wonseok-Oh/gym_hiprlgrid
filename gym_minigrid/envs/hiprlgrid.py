from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from gym_minigrid.envs.empty import EmptyEnv
import numpy as np
from SpatialMap import SpatialMap, ObjectMap
import rospy
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf import transformations
import time

from find_frontier.srv import InitPos, InitPosResponse

#import tf_conversions
#import tf2_ros
from math import pi

class HiPRLGridV0(MiniGridEnv):
    """
    Environment similar to kitchen.
    This environment has goals and rewards.
    """

    # Enumeration of possible actions
    class MetaActions(IntEnum):
        # explore, scan, plan
        explore = 0
        scan = 1
        plan = 2
        
        # stop this episode
        stop = 3
    

    
    def __init__(self, grid_size_ = 10, max_steps_ = 300, agent_view_size_ = 5, num_objects=3, num_boxes = 3):
        self.init_node = rospy.init_node('map_publisher', anonymous=True)
        self.spatial_map_pub = rospy.Publisher("spatial_map", OccupancyGrid, queue_size = 1, latch=True)
        self.object_map_pub = rospy.Publisher("object_map", OccupancyGrid, queue_size = 1)
        self.agent_pos_pub = rospy.Publisher("pose",PoseStamped, queue_size = 1, latch=True)
        self.agent_init_pos_pub = rospy.Publisher("initial_pose", PoseStamped, queue_size = 1, latch=True)
        self.navigation_map_pub = rospy.Publisher("navigation_map", OccupancyGrid, queue_size = 1, latch=True)
        self.explore_action_sub = rospy.Subscriber("action", Int32, self.explore_action_cb)
        # temporal abstraction: 5 timestep here (param)
        self.temp_abstr_lev = 5
        self.coverage_reward_coeff = 1.0/26
        self.spatial_map = SpatialMap(grid_size_, grid_size_)
        self.object_map = ObjectMap(grid_size_, grid_size_)
        self.prev_agent_pos = None
        self.num_objects = num_objects
        self.meta_actions = HiPRLGridV0.MetaActions
        self.num_boxes = num_boxes
        self.width = grid_size_
        self.height = grid_size_
        self.render_counter = 0
        self.explore_action = 0
        self.explore_action_set = False
        #self.br = tf2_ros.TransformBroadcaster()
        
        
        obs = super().__init__(grid_size = grid_size_, max_steps= max_steps_, agent_view_size = agent_view_size_)
        #print(obs)
        self.update_maps(obs, None)
        #print("agent_pos: %d, %d" %(self.agent_pos[0], self.agent_pos[1]))
        #print("agent_dir: %d" % self.agent_dir)
        #print("map_size: %d x %d" %(self.spatial_map.map.info.width, self.spatial_map.map.info.height))
#        for i in range(5):
#            self.spatial_map_pub.publish(self.spatial_map.map)
#            self.object_map_pub.publish(self.object_map.map)
#            self.spatial_map.rate.sleep()
    def explore_action_cb(self, msg):
        self.explore_action = msg.data
        self.explore_action_set = True


    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height-1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width-1, 0)

        # Types and colors of objects we can generate
        types = ['key', 'ball']

        objs = []
        objPos = []
        boxes = []
        boxPos = []

        def near_obj(env, p1):
            for p2 in objPos:
                dx = p1[0] - p2[0]
                dy = p1[1] - p2[1]
                if abs(dx) <= 1 and abs(dy) <= 1:
                    return True
            return False
        # initially, put a box containing a ball
        boxColor = self._rand_elem(COLOR_NAMES)
        objColor = self._rand_elem(COLOR_NAMES)
        obj = Box(len(objs), boxColor, Ball(len(objs)+1, objColor) )
        boxes.append(boxColor)
        objs.append(('ball', objColor))
        pos = self.place_obj(obj, reject_fn=near_obj)
        objPos.append(pos)
        boxPos.append(pos)
        # Until we have generated all the objects
        while len(objs) < self.num_objects:
            objType = self._rand_elem(types)
            objColor = self._rand_elem(COLOR_NAMES)

            # If this object already exists, try again
            if (objType, objColor) in objs:
                continue

            if objType == 'key':
                obj = Key(len(objs), objColor)
            elif objType == 'ball':
                obj = Ball(len(objs), objColor)
                
            pos = self.place_obj(obj, reject_fn=near_obj)

            objs.append((objType, objColor))
            objPos.append(pos)

        while len(boxes) < self.num_boxes:
            boxColor = self._rand_elem(COLOR_NAMES)
            
            # If this object already exists, try again
            if boxColor in boxes:
                continue
            
            box = Box(len(boxes), boxColor)
            #print("box.isOpen: %d" % box.isOpen)
            pos = self.place_obj(box, reject_fn=near_obj)
            boxes.append(boxColor)
            boxPos.append(pos)
        # Randomize the agent start position and orientation
        self.place_agent()

        # Choose a random object to be moved
        self.objIdx = self._rand_int(0, len(objs))
        self.move_type, self.moveColor = objs[self.objIdx]
        self.move_pos = objPos[self.objIdx]

        # Choose a target object (to put the first object into)
        self.targetIdx = self._rand_int(0, len(boxes))
        self.target_color = boxes[self.targetIdx]
        self.target_pos = boxPos[self.targetIdx]

        self.mission = 'put the %s %s into the %s %s' % (
            self.moveColor,
            self.move_type,
            self.target_color,
            'box'
        )
        
    def reset(self):
        #print("reset is called")
        self.opened_receptacles = set()
        self.closed_receptacles = set()
        self.seen_obj = set()
        self.seen_box = set()
        self.checked_receptacles = set()
        self.visited_locations = set()
        self.can_end = False
        self.object_map = ObjectMap(self.width, self.height)
        self.spatial_map = SpatialMap(self.width, self.height)
        self.new_coverage = 0
        self.prev_agent_pos = None
        self.render_counter = 0
        return super().reset()
    
    def step(self, action):
        
        preCarrying = self.carrying
        #print("agent_pos: %d, %d" %(self.agent_pos[0], self.agent_pos[1]))
        #print("agent_dir: %d" % self.agent_dir)
        obs, reward, done, info = MiniGridEnv.step(self, action)
        reward = -0.01 # constant time penalty

        # update spatial map & object map
        # self.update_maps(obs, action)
        #self.spatial_map_pub.publish(self.spatial_map.map)
        #self.object_map_pub.publish(self.object_map.map)
        #self.spatial_map.rate.sleep()
        self.update_maps(obs, action)
        self.spatial_map_pub.publish(self.spatial_map.map)
        self.object_map_pub.publish(self.object_map.map)
        self.publish_ros_agent_pos()
        #self.broadcast_tf()

        self.spatial_map.rate.sleep()
        # reward for open/close action
        if info is not None:
            if info.can_toggle() and action == self.actions.open:
                if info.objectId in self.closed_receptacles: # if the receptacle was closed
                    self.opened_receptacles.add(info.objectId) # add to the opened_receptacles list
                    self.closed_receptacles.discard(info.objectId)
                    if info.objectId in self.checked_receptacles: # if the receptacle was checked before, penalize
                        reward += -1.0
                    else:                                    # else, if it was not checked, give reward
                        self.checked_receptacles.add(info.objectId)
                        reward += 1.0
                
                elif info.objectId in self.opened_receptacles: # penalty for open opened_receptacles
                    reward += -1.0
            elif info.can_toggle() and action == self.actions.close:            
                if info.objectId in self.opened_receptacles: # if the receptacle was opened
                    self.closed_receptacles.add(info.objectId) # add to the closed_receptacles list
                    self.opened_receptacles.discard(info.objectId)
                    
                elif info.objectId in self.closed_receptacles:
                    reward += -1.0 # penalty for close closed_receptacles    
        
        
        # reward(penalize) for carrying (non)target object or 
        # just finish the episode
        if self.carrying != None:
            if self.objIdx == self.carrying.objectId:
                reward += 1.0
            else:
                reward += -1.0

        # If successfully dropping an object into the target
        u, v = self.dir_vec
        ox, oy = (self.agent_pos[0] + u, self.agent_pos[1] + v)
        tx, ty = self.target_pos
        if action == self.actions.drop and preCarrying:
            front_obj = self.grid.get(ox,oy)
            if front_obj.type is 'box' and front_obj.contains is preCarrying:
                if abs(ox - tx) == 0 and abs(oy - ty) == 0:
                    done = True
                    reward += 10
        
        # coverage reward
        reward += self.new_coverage * self.coverage_reward_coeff
        
        self.prev_agent_pos = self.agent_pos
        self.prev_agent_dir = self.agent_dir
        
        #print(self.opened_receptacles)
        #print(self.closed_receptacles)
        #print(self.checked_receptacles)

        return obs, reward, done, info

    def invoke(self, meta_action):
        if meta_action == self.meta_actions.explore:
            self.explore()
        elif meta_action == self.meta_actions.plan:
            self.plan()
        elif meta_action == self.meta_actions.scan:
            self.scan()
        elif meta_action == self.meta_actions.stop:
            self.stop()
        else:
            print('this meta_action should not happen')
    
    def render(self, mode = 'human', close = False, highlight = True, tile_size = TILE_PIXELS):
        img = super().render(mode, close, highlight, tile_size)
        obs = self.gen_obs()
        if self.render_counter == 0:
            self.update_maps(obs, None)
            self.spatial_map_pub.publish(self.spatial_map.map)
            self.object_map_pub.publish(self.object_map.map)
            self.agent_init_pos = self.publish_ros_agent_pos()
            self.agent_init_dir = self.agent_dir
            self.agent_init_pos_pub.publish(self.agent_init_pos)
            #self.broadcast_tf()
        self.render_counter += 1
        return img
    '''
    def broadcast_tf(self):
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "map"
        t.child_frame_id = "base_link"
        t.transform.translation.x = (self.agent_pos[0] + 0.5) * self.spatial_map.map.info.resolution
        t.transform.translation.y = (self.agent_pos[1] + 0.5) * self.spatial_map.map.info.resolution
        t.transform.translation.z = 0
        q = tf_conversions.transformations.quaternion_from_euler(0, 0, self.agent_dir*pi/2 + pi/2)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        br.sendTransform(t)
    '''    
    def publish_ros_agent_pos(self):
        ros_agent_pos = PoseStamped()
        ros_agent_pos.header.frame_id = "map"
        ros_agent_pos.header.stamp = rospy.Time.now()
        ros_agent_pos.pose.position.x = (self.agent_pos[0]+0.5) * self.spatial_map.map.info.resolution 
        ros_agent_pos.pose.position.y = (self.spatial_map.map.info.height - self.agent_pos[1] - 1 + 0.5) * self.spatial_map.map.info.resolution
        ros_agent_pos.pose.position.z = 0
        orientation = self.dir_to_quaternion()
        ros_agent_pos.pose.orientation.x = orientation[0]
        ros_agent_pos.pose.orientation.y = orientation[1]
        ros_agent_pos.pose.orientation.z = orientation[2]
        ros_agent_pos.pose.orientation.w = orientation[3]
        self.agent_pos_pub.publish(ros_agent_pos)
        return ros_agent_pos

    def dir_to_quaternion(self):
        dir = self.agent_dir
        ##########
        # 0: >
        # 1: V
        # 2: <
        # 3: ^
        ##########
        if dir == 0:
            yaw = 0
        elif dir == 1:
            yaw = -pi/2
        elif dir == 2:
            yaw = pi
        elif dir == 3:
            yaw = pi/2
        return transformations.quaternion_from_euler(0, 0, yaw)      
    
    # now, spatial map update algorithm is done    
    def update_maps(self, obs, action):
        if np.array_equal(self.prev_agent_pos, self.agent_pos) and np.array_equal(self.prev_agent_dir, self.agent_dir): # no movement
            if action == self.actions.open or action == self.actions.pickup or action == self.actions.drop:
                front = self.front_pos
                fwd_cell = self.grid.get(front[0], front[1])

                front[1] = self.spatial_map.map.info.height - front[1] - 1
                if fwd_cell is None: # empty cell or out of range
                    if 0 <= front[0] and front[0] <= self.spatial_map.map.info.width-1 and 0 <= front[1] and front[1] <= self.spatial_map.map.info.height-1:
                        self.spatial_map.update_cell(front, SpatialMap.OccGridStates.free)
                        self.object_map.update_cell(front, ObjectMap.ObjGridStates.floor)
                        
                else:
                    
                    # update object map
                    if fwd_cell.type == 'box':
                        print("fwd_cell.objectId: %d" % fwd_cell.objectId)
                        if fwd_cell.isOpen == True:
                            if fwd_cell.contains is not None:
                                if fwd_cell.contains.type == 'ball':
                                    self.object_map.update_cell(front, ObjectMap.ObjGridStates.ball, center_only = True)
                                elif fwd_cell.contains.type == 'key':
                                    self.object_map.update_cell(front, ObjectMap.ObjGridStates.key, center_only = True)
                            else:
                                self.object_map.update_cell(front, ObjectMap.ObjGridStates.box)
                        
                        elif fwd_cell.objectId not in self.checked_receptacles:
                            self.object_map.update_cell(front, ObjectMap.ObjGridStates.box)
                        
                    
                    elif fwd_cell.type == 'key':
                        self.object_map.update_cell(front, ObjectMap.ObjGridStates.key, center_only = True)
                        
                    elif fwd_cell.type == 'ball':
                        self.object_map.update_cell(front, ObjectMap.ObjGridStates.ball, center_only = True)
                    
                    elif fwd_cell.type == 'wall':
                        self.object_map.update_cell(front, ObjectMap.ObjGridStates.wall)
                    
                    else:
                        print("update_maps: this should not happen. new type")
                
                    # update spatial map
                    if fwd_cell.can_overlap():
                        self.spatial_map.update_cell(front, SpatialMap.OccGridStates.free)
                    else:
                        self.spatial_map.update_cell(front, SpatialMap.OccGridStates.occupied)
    
                        
            else:
                return
        else:
            # for all the cells in the agent's view, update the cells info into the map unless it is not chekced receptacles
            obs_grid, _ = Grid.decode(obs['image'])
            for i in range(obs_grid.width):
                for j in range(obs_grid.height):
                    object = obs_grid.get(i,j)
                    wx, wy = self.get_world_coordinate(i,j)
                    if np.array_equal(self.agent_pos, np.array([wx,wy])):
                        wy = self.spatial_map.map.info.height - wy - 1 
                        self.spatial_map.update_cell(np.array([wx,wy]), SpatialMap.OccGridStates.free)
                        continue
                    wy = self.spatial_map.map.info.height - wy - 1 

                    if object is None: # empty cell or out of range
                        if 0 <= wx and wx <= self.spatial_map.map.info.width-1 and 0 <= wy and wy <= self.spatial_map.map.info.height-1:
                            self.spatial_map.update_cell(np.array([wx,wy]), SpatialMap.OccGridStates.free)
                            self.object_map.update_cell(np.array([wx,wy]), ObjectMap.ObjGridStates.floor)
                            
                    else:
                        # update object map
                        if object.type == 'box':
                            if object.objectId in self.checked_receptacles:
                                pass
                            
                            elif object.isOpen == True:
                                self.opened_receptacles.add(object.objectId)
                                if object.contains is not None:
                                    if object.contains.type == 'ball':
                                        self.object_map.update_cell(np.array([wx,wy]), ObjectMap.ObjGridStates.ball, center_only = True)
                                    elif object.contains.type == 'key':
                                        self.object_map.update_cell(np.array([wx,wy]), ObjectMap.ObjGridStates.key, center_only = True)
                                else:
                                    self.object_map.update_cell(np.array([wx,wy]), ObjectMap.ObjGridStates.box)
                            
                            else:
                                self.closed_receptacles.add(object.objectId)
                                self.object_map.update_cell(np.array([wx, wy]), ObjectMap.ObjGridStates.box)
                                
                            
                        elif object.type == 'key':
                            self.object_map.update_cell(np.array([wx,wy]), ObjectMap.ObjGridStates.key, center_only = True)
                            
                        elif object.type == 'ball':
                            self.object_map.update_cell(np.array([wx,wy]), ObjectMap.ObjGridStates.ball, center_only = True)
                        
                        elif object.type == 'wall':
                            self.object_map.update_cell(np.array([wx, wy]), ObjectMap.ObjGridStates.wall)
                        
                        else:
                            print("update_maps: this should not happen. new type")
                            
                        # update spatial map
                        if object.can_overlap():
                            self.spatial_map.update_cell(np.array([wx,wy]), SpatialMap.OccGridStates.free)
                        else:
                            self.spatial_map.update_cell(np.array([wx,wy]), SpatialMap.OccGridStates.occupied)
                
        self.spatial_map.map.header.stamp = rospy.Time.now()
        self.object_map.map.header.stamp = rospy.Time.now()                   

    def scan(self):
        self.step(self.Actions.left)
        self.step(self.Actions.left)
        self.step(self.Actions.left)
        self.step(self.Actions.left)

    def explore(self):
        rospy.wait_for_service('init_pose_update')
        result = False
        try:
            init_pose_update = rospy.ServiceProxy('init_pose_update', InitPos)
            result = init_pose_update(self.agent_init_pos.pose.position.x, self.agent_init_pos.pose.position.y, self.agent_init_dir)
        except rospy.ServiceException as e:
            print("Service call failed: %s" %e)
        print("init_pose_update: ", result)

        try:
            pose_update = rospy.ServiceProxy('pose_update', InitPos)
            x = (self.agent_pos[0] + 0.5) * self.spatial_map.map.info.resolution
            y = (self.spatial_map.map.info.height - self.agent_pos[1] - 0.5) * self.spatial_map.map.info.resolution
            result = pose_update(x, y, self.agent_dir)
        except rospy.ServiceException as e:
            print("Service call failed: %s" %e)
        
        print("pose_update: ", result)       
    
        # Execute mapConvert in findFrontier, which generates plan
        print("action: ", self.explore_action)
        self.navigation_map_pub.publish(self.spatial_map.map)
        while(self.explore_action_set is not True):
            time.sleep(0.1)
            #block
        print("action: ", self.explore_action)
        self.explore_action_set = False
        return self.explore_action
            
register(
    id='MiniGrid-HiPRLGrid-v0',
    entry_point='gym_minigrid.envs:HiPRLGridV0'
)

""" For advanced sim
        # Place random objects in the world
        types1 = ['cup', 'apple', 'egg']
        types2 = ['cabinet', 'table', 'refrigerator', 'microwave', 'sink']
        objColor = self._rand_elem(COLOR_NAMES)
        tableObj = Table(objColor)
        self.place_obj(tableObj)
        
        objColor = self._rand_elem(COLOR_NAMES)
        refrigeratorObj = Refrigerator(objColor)
        self.place_obj(refrigeratorObj)
        
        objColor = self._rand_elem(COLOR_NAMES)
        microwaveObj = Microwave(objColor)
        self.place_obj(microwaveObj)
        
        objColor = self._rand_elem(COLOR_NAMES)
        sinkObj = Sink(objColor)
        self.place_obj(sinkObj)
        
        objColor = self._rand_elem(COLOR_NAMES)
        cabinetObj = Cabinet(objColor)
        self.place_obj(cabinetObj)
            
        for i in range(0, 5):
            objType = self._rand_elem(types1)
            objColor = self._rand_elem(COLOR_NAMES)
            if objType == 'cup':
                obj = Cup(i, objColor)
            elif objType == 'apple':
                obj = Apple(i, objColor)
            elif objType == 'egg':
                obj = Egg(i, objColor)         
            self.place_obj(obj)

        # No explicit mission in this environment --> to be modified
        self.mission = '' """