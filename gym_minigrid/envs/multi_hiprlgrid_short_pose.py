from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from gym_minigrid.envs.empty import EmptyEnv
import numpy as np
from SpatialMap import SpatialMap, ObjectMap, BinaryMap, BinaryMap_MLP, ObjectMap_MLP
import rospy
from std_srvs.srv import Empty
from rosplan_dispatch_msgs.srv import DispatchService
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Int32MultiArray
from find_frontier.msg import ActionArray
from geometry_msgs.msg import PoseStamped, TransformStamped, PoseArray, Pose
from tf import transformations
import time, copy
import roslaunch
from rosparam import upload_params
from yaml import load

from find_frontier.srv import *
from hiprl_replicate.msg import Obs, Obs_multi
from hiprl_replicate.srv import ActionExecution, ActionExecutionRequest
from rosplan_dispatch_msgs.msg import CompletePlan
from rosplan_knowledge_msgs.msg import processReset

#import tf_conversions
#import tf2_ros
from math import pi
#from builtins import None



class MultiHiPRLGridShortPoseV0(MiniGridEnv):
    """
    Environment similar to kitchen.
    This environment has goals and rewards.
    """
    def dir_vec_i(self, id):
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.
        """

        assert self.agents[id].dir >= 0 and self.agents[id].dir < 4
        return DIR_TO_VEC[self.agents[id].dir]

    def right_vec_i(self, id):
        """
        Get the vector pointing to the right of the agent.
        """

        dx, dy = self.dir_vec_i(id)
        return np.array((-dy, dx))
    
    class Agent(object):
        def __init__(self, id, multihiprlgridshortposev0):
            self.multihiprlgridshortposev0 = multihiprlgridshortposev0
            self.mode = self.multihiprlgridshortposev0.Option_mode.init
            self.pos = None
            self.prev_pos = None
            self.dir = None
            self.prev_dir = None
            self.id = id
            self.action_list = []
            self.carrying = None
            self.preCarrying = None

        def front_pos(self):
            assert self.dir>= 0 and self.dir < 4
            return self.pos + DIR_TO_VEC[self.dir]

    # Enumeration of possible actions
    class MetaActions(IntEnum):
        # explore, scan, plan
        explore = 0
        scan = 1
        plan = 2
        #keep_previous = 3
        # stop this episode
        #stop = 3
    
    class Option_mode(IntEnum):
        init = 0
        explore = 1
        scan = 2
        plan = 3    
    
    class PlanningMode(IntEnum):
        search = 0
        keep = 1
        bring = 2
    
    def place_agent_i(
        self,
        id = 0,
        top=None,
        size=None,
        rand_dir=True,
        max_tries=math.inf
    ):
        """
        Set the agent's starting point at an empty position in the grid
        """

        self.agents[id].pos = None
        pos = self.place_obj_multi(None, top, size, max_tries=max_tries)
        self.agents[id].pos = pos

        if rand_dir:
            self.agents[id].dir = self._rand_int(0, 4)
        return pos
    
    def place_obj_multi(self,
        obj,
        top=None,
        size=None,
        reject_fn=None,
        max_tries=math.inf
    ):
        """
        Place an object at an empty position in the grid

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
#            if obj is not None:
#                print('placing object %s' %(obj.type))
                
            if num_tries > max_tries:
                raise RecursionError('rejection sampling failed in place_obj')

            num_tries += 1

            pos = np.array((
                self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                self._rand_int(top[1], min(top[1] + size[1], self.grid.height))
            ))

            # Don't place the object on top of another object
            if self.grid.get(*pos) != None:
                continue

            obj_agent_same_pos = False
            # Don't place the object where the agent is
            for agent in self.agents:
                if np.array_equal(pos, agent.pos):
                    obj_agent_same_pos = True
                    break
            if obj_agent_same_pos:
                continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break

        self.grid.set(*pos, obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos
        
        
    
    def __init__(self, grid_size_ = 20, max_steps_ = 200, agent_view_size_ = 5, num_objects=1, num_boxes = 10, process_num = 0, num_agents=3):
        self.agents = [self.Agent(i, self) for i in range(num_agents)]
        self.num_agents = num_agents
        see_through_walls = False
        seed = 1337
        self.process_num = process_num
        self.mode = self.Option_mode.init

        width = None
        height = None

        # Can't set both grid_size and width/height
        if grid_size_:
            assert width == None and height == None
            width = grid_size_
            height = grid_size_

        # Action enumeration for this environment
        self.actions = MiniGridEnv.Actions

        # Actions are discrete integer values
        self.action_space = spaces.MultiDiscrete([len(self.actions), len(self.actions), len(self.actions)])
        
        # Number of cells (width and height) in the agent view
        assert agent_view_size_ % 2 == 1
        assert agent_view_size_ >= 3
        self.agent_view_size = agent_view_size_

        # Meta action enumeration for this environment
        self.meta_actions = self.MetaActions
        
        # Meta actions are discrete integer values
        self.meta_action_space = spaces.MultiDiscrete([len(self.meta_actions), len(self.meta_actions), len(self.meta_actions)])
        
        # total action space & meta action space
        #self.total_action_space = [self.action_space for i in range(num_agents)]
        #self.total_meta_action_space = [self.meta_action_space for i in range(num_agents)]

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(width*height*10+3,),
            dtype='uint8'
        )

        # variable to check whether the mission complete or not
        self.success = False
        print('gym_env' + str(process_num))

        # ROS Environment for communication with external process (explorer, knowledge manager, planner)
        if (process_num % 8 == 0):
            self.init_node = rospy.init_node('gym_env' + str(process_num), anonymous=True)
        self.spatial_map_pub = rospy.Publisher("spatial_map" + str(process_num), OccupancyGrid, queue_size = 1, latch=True)
        self.object_map_pub = rospy.Publisher("object_map" + str(process_num), OccupancyGrid, queue_size = 1)
        self.agent_pos_pub = rospy.Publisher("pose" + str(process_num), PoseArray, queue_size = 1, latch=True)

        self.agents_pose_pub_vis = []
        for i in range(num_agents):
            self.agents_pose_pub_vis.append(rospy.Publisher("pose" + str(process_num) +'_'+ str(i), PoseStamped, queue_size = 1, latch=True))
        self.goal_pos_pub = rospy.Publisher("goal_pose" + str(process_num), PoseStamped, queue_size = 1, latch=True)
        self.agent_init_pos_pub = rospy.Publisher("initial_pose" + str(process_num), PoseArray, queue_size = 1, latch=True)
        self.navigation_map_pub = rospy.Publisher("navigation_map" + str(process_num), OccupancyGrid, queue_size = 1, latch=True)
        self.planning_map_pub = rospy.Publisher("planning_map" + str(process_num), OccupancyGrid, queue_size = 1, latch=True)
        self.explore_action_sub = rospy.Subscriber("action_plan" + str(process_num), ActionArray, self.explore_plan_cb)
        self.observation_pub = rospy.Publisher("observation" + str(process_num), Obs_multi, queue_size = 1)
        self.reset_pub = rospy.Publisher("rosplan_knowledge_base" + str(process_num)+ "/reset", processReset, queue_size = 1, latch=True)
        #self.action_service = rospy.Service("action_execution", ActionExecution, self.execute_action)
        self.complete_plan_sub = rospy.Subscriber("complete_plan"+ str(process_num), ActionArray, self.complete_plan_cb)
        #self.complete_plan_sub = []
        #for i in range(num_agents):
        #    self.complete_plan_sub.append(rospy.Subscriber("/rosplan_parsing_interface"+ str(process_num) + "_" + str(i) + "/complete_plan" , CompletePlan, self.complete_plan_cb, (i)))

        # Reward Coefficients
        self.coverage_reward_coeff = 0.00002
        self.open_reward_coeff = 0.01
        self.carry_reward_coeff = 0.2
        
        # Map initialize
        self.spatial_map = SpatialMap(grid_size_, grid_size_)
        self.object_map = ObjectMap(grid_size_, grid_size_, 3)
        
        # Planning mode init
        self.planning_mode = self.PlanningMode.search
        
        #self.floor_map_one_hot = BinaryMap(grid_size_, grid_size_)
        #self.goal_map_one_hot = BinaryMap(grid_size_, grid_size_)
        #self.wall_map_one_hot = BinaryMap(grid_size_, grid_size_)
        #self.box_map_one_hot = BinaryMap(grid_size_, grid_size_)
        #self.checked_box_map_one_hot = BinaryMap(grid_size_, grid_size_)
        #self.ball_map_one_hot = BinaryMap(grid_size_, grid_size_)
        #self.unknown_map_one_hot = BinaryMap(grid_size_, grid_size_)

        self.floor_map_one_hot = BinaryMap_MLP(grid_size_, grid_size_)
        self.goal_map_one_hot = BinaryMap_MLP(grid_size_, grid_size_)
        self.wall_map_one_hot = BinaryMap_MLP(grid_size_, grid_size_)
        self.box_map_one_hot = BinaryMap_MLP(grid_size_, grid_size_)
        self.checked_box_map_one_hot = BinaryMap_MLP(grid_size_, grid_size_)
        self.ball_map_one_hot = BinaryMap_MLP(grid_size_, grid_size_)
        self.unknown_map_one_hot = BinaryMap_MLP(grid_size_, grid_size_)
        
        self.num_objects = num_objects
        self.num_boxes = num_boxes
        self.width = grid_size_
        self.height = grid_size_
        self.render_counter = 0
        self.explore_action_list = []
        self.explore_action_set = False
        self.complete_plan = [] * self.num_agents
        self.complete_plan_flag = False
        self.complete_plan_id_list = set([])
        self.goal_pos = None
        self.planner_action_num = 0

        # Environment configuration
        self.width = width
        self.height = height
        self.max_steps = max_steps_
        self.see_through_walls = see_through_walls

        self.seed(seed = seed)
        self.reset()
        #return self.reset()
        #print(obs)
        #self.update_maps(obs, None)
        #print("agent_pos: %d, %d" %(self.agent_pos[0], self.agent_pos[1]))
        #print("agent_dir: %d" % self.agent_dir)
        #print("map_size: %d x %d" %(self.spatial_map.map.info.width, self.spatial_map.map.info.height))
#        for i in range(5):
#            self.spatial_map_pub.publish(self.spatial_map.map)
#            self.object_map_pub.publish(self.object_map.map)
#            self.spatial_map.rate.sleep()

    #def __del__(self):
        #self.process0.stop()
        #self.process1.stop()
        #self.process2.stop()
        #self.process3.stop()
        #self.process4.stop()
        #self.process5.stop()
        #self.process6.stop()
        #self.launch.stop()
        #rospy.signal_shutdown("End class")

    def complete_plan_cb(self, msg):
        self.complete_plan_flag = True
        print("complete_plan: {}".format(msg))

    def generate_actions_from_complete_plan(self, id_list):
        potential_goal_update = False
        planner_tracking_dir = [-1] * self.num_agents
        for i in id_list:
            planner_tracking_dir[i] = self.agents[i].dir
            self.agents[i].action_list = []
             
        prev_time = 0
        actions = []
        for item in self.complete_plan:
            item_actions = []
            prev_time = item_time
            item_time = item.dispatch_time * 1000
            
            # if one timestep passed, let rest of agent do nothing
            if item_time - prev_time >= 1:
                current_max_len = 0
                for i in id_list:
                    if len(self.agents[i].action_list) > current_max_len:
                        current_max_len = len(self.agents[i].action_list)
                
                for i in id_list:
                    if current_max_len > len(self.agents[i].action_list):
                        for j in range(current_max_len - len(self.agents[i].action_list)):
                            self.agents[i].action_list.append(self.Actions.done)
                
            agent_key_value = item.parameters[0]
                
            # the robot parameter should be set as 'a'
            assert(agent_key_value.key == 'a')

            agent_num = agent_key_value.value
            #print('agent: {}'.format(agent_num))
            agent_num = int(agent_num[-1])
            #print('agent_num: {}'.format(agent_num))
            
            dir_key_value = item.parameters[-1]
        
            # the last parameter should set key as dir
            assert(dir_key_value.key == 'dir')
        
            # make robot to face given direction
            if dir_key_value.value == 'left':
                goal_dir = 2
            elif dir_key_value.value == 'up':
                goal_dir = 3
            elif dir_key_value.value == 'right':
                goal_dir = 0
            elif dir_key_value.value == 'down':
                goal_dir = 1

            dth = planner_tracking_dir[agent_num] - goal_dir
            if dth < 0:
                dth += 4
            if dth == 3:
                item_actions.append(self.Actions.right)
            
            else:
                for i in range(dth):
                    item_actions.append(self.Actions.left)
            
            planner_tracking_dir[agent_num] = goal_dir
            if item.name == "move-robot":
                item_actions.append(self.Actions.forward)
                
    
            elif item.name == "openobject":
                item_actions.append(self.Actions.open)
                potential_goal_update = True
                
            elif item.name == "pickupobjectinreceptacle" or item.name == "pickupobject":
                item_actions.append(self.Actions.pickup)
                potential_goal_update = True
                
            elif item.name == "putobjectinreceptacle" or item.name == "putobject":
                item_actions.append(self.Actions.drop)
                
            elif item.name == "closeobject":
                item_actions.append(self.Actions.close)
            
            actions.append((item, item_actions, potential_goal_update))
            
            for action in item_actions:
                self.agents[agent_num].action_list.append(action)
            
        return actions
        
    def explore_plan_cb(self, msg):
        print("Explore_plan_cb called")
        j = 0
        min_len = 1000
        for i in range(self.num_agents):
            if self.agents[i].mode != self.Option_mode.explore:
                continue
            
            if j >= len(msg.list):
                break
            
            #print("len(msg.list): {}, j: {}".format(len(msg.list), j))
            if len(msg.list) > 0 and len(msg.list[j].data) < min_len and len(msg.list[j].data) > 1:
                min_len = len(msg.list[j].data) 
            #print("agent {}'s action length: {}".format(i, len(msg.list[j].data)))
            self.agents[i].action_list  = [0] * len(msg.list[j].data)
            for k in range(len(msg.list[j].data)):
                self.agents[i].action_list[k] = msg.list[j].data[k]
            
            j = j + 1
        
        j = 0
        for i in range(self.num_agents):
            if self.agents[i].mode != self.Option_mode.explore:
                continue
            
            if j >= len(msg.list):
                break
            
            if len(msg.list) > 0 and len(msg.list[j].data) <= 1:
                if len(msg.list[j].data) == 0 or msg.list[j].data[0] < 0:
                #self.agents[i].action_list  = [self.Actions.done] * min_len
                    self.meaningless = True
            j = j + 1
        
        if min_len == 1000:
            for i in range(self.num_agents):
                if self.agents[i].mode != self.Option_mode.explore:
                    continue
                #self.agents[i].action_list = [self.Actions.done]
                self.meaningless = True
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
        self.boxPos = []

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
        obj = Box(len(boxes), boxColor, Ball(len(objs), objColor) )
        boxes.append(boxColor)
        objs.append(('ball', objColor))
        pos = self.place_obj_multi(obj, reject_fn=near_obj)
        objPos.append(pos)
        self.boxPos.append(pos)
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
                
            pos = self.place_obj_multi(obj, reject_fn=near_obj)

            objs.append((objType, objColor))
            objPos.append(pos)

        while len(boxes) < self.num_boxes:
            boxColor = self._rand_elem(COLOR_NAMES)
            
            # If this object already exists, try again
            #if boxColor in boxes:
            #    continue
            
            box = Box(len(boxes), boxColor)
            #print("box.isOpen: %d" % box.isOpen)
            pos = self.place_obj_multi(box, reject_fn=near_obj)
            boxes.append(boxColor)
            self.boxPos.append(pos)
        
        # place a goal    
        obj = Goal()
        objColor = self._rand_elem(COLOR_NAMES)
        self.goal_pos = self.place_obj_multi(obj, reject_fn = near_obj)
        
        # publish the goal position to update the knowledge base
        self.publish_ros_goal_pos(self.goal_pos)
        
        # Randomize the agent start position and orientation
        for i in range(len(self.agents)):
            self.place_agent_i(i)

        # Choose a random object to be moved
        self.objIdx = self._rand_int(0, len(objs))
        self.move_type, self.moveColor = objs[self.objIdx]
        self.move_pos = objPos[self.objIdx]

        # Choose a target object (to put the first object into)
        self.target_pos = self.goal_pos

        self.mission = 'put the %s %s into the goal' % (
            self.moveColor,
            self.move_type,
        )
        
    def reset(self):
        #print("reset is called")
        for agent in self.agents:
            agent.pos = None
            agent.dir = None
            
            # Item picked up, being carried, initially nothing
            agent.carrying = None
            agent.preCarrying = None
            agent.mode = agent.multihiprlgridshortposev0.Option_mode.init
            agent.prev_pos = None
            agent.prev_dir = None
            agent.action_list = []
            
            
        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid(self.width, self.height)
        
        # These fields should be defined by _gen_grid
        for agent in self.agents:
            #print("agent.pos: {}".format(agent.pos))
            #print("agent.dir: {}".format(agent.dir))

            assert agent.pos is not None
            assert agent.dir is not None
        
        # Check that the agent doesn't overlap with an object
        for agent in self.agents:
            start_cell = self.grid.get(*agent.pos)
            assert start_cell is None or start_cell.can_overlap()
        
        # Step count since episode start
        self.step_count = 0
                
        self.opened_receptacles = set()
        self.closed_receptacles = set()
        self.seen_obj = set()
        self.seen_receptacles = set()
        self.checked_receptacles = set()
        self.visited_locations = set()
        self.can_end = False
        self.object_map = ObjectMap(self.width, self.height, 3)
        self.spatial_map = SpatialMap(self.width, self.height)
        self.object_map_mlp = ObjectMap_MLP(self.width, self.height, ObjectMap_MLP.ObjGridStates.unknown)
        
        self.floor_map_one_hot = BinaryMap_MLP(self.width, self.height)
        self.goal_map_one_hot = BinaryMap_MLP(self.width, self.height)
        self.wall_map_one_hot = BinaryMap_MLP(self.width, self.height)
        self.box_map_one_hot = BinaryMap_MLP(self.width, self.height)
        self.checked_box_map_one_hot = BinaryMap_MLP(self.width, self.height)
        self.ball_map_one_hot = BinaryMap_MLP(self.width, self.height)
        self.unknown_map_one_hot = BinaryMap_MLP(self.width, self.height, value = 1)
        
        self.planning_mode = self.PlanningMode.search
        
        self.new_coverage = 0
        self.prev_agent_pos = None
        self.render_counter = 0
        self.explore_action_set = False
        #self.planner_action_set = False
        #self.planner_action_plan = []
        self.planner_action_num = 0
        #self.dispatch_plan_end = True
        self.action_execution_flag = False
        self.complete_plan = []
        self.complete_plan_flag = False
        reset_msg = processReset()
        reset_msg.domain_path = "/home/morin/catkin_ws/src/hiprl_replicate/pddl/hiprl_mini_multi.pddl"
        reset_msg.problem_path = "/home/morin/catkin_ws/src/hiprl_replicate/pddl/hiprl_problem0_multi.pddl"
        self.reset_pub.publish(reset_msg)
        obs = self.gen_obs()
        
        # Temporal comment for test
        self.update_maps(obs, None)
        self.spatial_map_pub.publish(self.spatial_map.map)
        self.object_map_pub.publish(self.object_map.map)
        self.agent_init_pos = self.publish_ros_agent_pos()
        self.agent_init_dir = [self.agents[i].dir for i in range(len(self.agents))] 
        self.agent_init_pos_pub.publish(self.agent_init_pos)
        self.publish_observation(obs['image'])
        self.success = False
        self.meaningless = False
        #return obs['image']
        
        return self.generate_network_input_one_hot_for_mlp(obs)
        #return np.reshape(self.object_map.map.data, (self.width*self.object_map.factor, self.height*self.object_map.factor))
    
    # Currently, collision is not considered yet --> should be considered
    def step(self, action):
        assert len(action) == self.num_agents
        
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        
        self.step_count += 1
        reward = 0
        done = False
        for i in range(len(self.agents)):
            info = None
            self.agents[i].preCarrying = self.agents[i].carrying
            self.agents[i].prev_pos = self.agents[i].pos
            self.agents[i].prev_dir = self.agents[i].dir
            fwd_pos = self.agents[i].front_pos()
            fwd_cell = self.grid.get(*fwd_pos)
            
            # Rotate left
            if action[i] == self.actions.left:
                self.agents[i].dir -= 1
                if self.agents[i].dir < 0:
                    self.agents[i].dir += 4
            
            # Rotate right
            elif action[i] == self.actions.right:
                self.agents[i].dir = (self.agents[i].dir + 1) % 4
                
            # Move forward
            elif action[i] == self.actions.forward:
                if fwd_cell == None or fwd_cell.can_overlap():
                    self.agents[i].pos = fwd_pos
            
            # Pick up an object
            elif action[i] == self.actions.pickup:
                if fwd_cell and fwd_cell.can_pickup():
                    if self.agents[i].carrying is None and fwd_cell.type != 'box':
                        self.agents[i].carrying = fwd_cell
                        self.agents[i].carrying.cur_pos = np.array([-1, -1])
                        self.grid.set(*fwd_pos, None)
                    elif self.agents[i].carrying is None and fwd_cell.type == 'box':
                        self.plannnig_mode = self.PlanningMode.bring
                        self.agents[i].carrying = fwd_cell.contains
                        self.agents[i].carrying.cur_pos = np.array([-1, -1])
                        self.grid.set(*fwd_pos, Box(fwd_cell.objectId, fwd_cell.color, contains = None, isOpen=True))
            
            # Drop an object                
            elif action[i] == self.actions.drop:
                if not fwd_cell:
                    if self.agents[i].carrying:
                        self.grid.set(*fwd_pos, self.agents[i].carrying)
                        self.agents[i].carrying.cur_pos = fwd_pos
                        self.agents[i].carrying = None
                elif fwd_cell.type == 'box' and fwd_cell.isOpen and fwd_cell.contains == None and self.agents[i].carrying:
                    fwd_cell.contains = self.agents[i].carrying
                    self.agents[i].carrying.cur_pos = fwd_pos
                    self.agents[i].carrying = None
                    
            # open an object
            elif action[i] == self.actions.open:
                if fwd_cell and fwd_cell.can_toggle():
                    fwd_cell.isOpen = True
                    info = fwd_cell
                    
            # close an object
            elif action[i] == self.actions.close:
                if fwd_cell and fwd_cell.can_toggle():
                    fwd_cell.isOpen = False
                    info = fwd_cell
                    
            # Done action (not used by default)
            elif action[i] == self.actions.done:
                pass

            else:
                assert False, "unknown action"

            info_n['fwd_cell'+str(i)] = info


        if self.step_count >= self.max_steps:
            done = True

                
        #print("agent_pos: %d, %d" %(self.agent_pos[0], self.agent_pos[1]))
        #print("agent_dir: %d" % self.agent_dir)
    
        #obs, reward, done, info = MiniGridEnv.step(self, action)
        obs = self.gen_obs()
        
        reward = -0.0001 # constant time penalty

        # update spatial map & object map
        #print("obs: {}".format(obs))
        self.update_maps(obs, action)
        self.spatial_map_pub.publish(self.spatial_map.map)
        self.object_map_pub.publish(self.object_map.map)
        self.publish_ros_agent_pos()
        
        # publish observation information to update knowledgebase
        self.publish_observation(obs['image'])
        
        #print(obs['image'])
        
        # reward for open/close action
        for i in range(len(self.agents)):
            #print(info)            
            if info_n['fwd_cell'+str(i)] is not None:
                if info_n['fwd_cell'+str(i)].can_toggle() and action[i] == self.Actions.open:
                    if info_n['fwd_cell'+str(i)].objectId in self.closed_receptacles: # if the receptacle was closed
                        self.opened_receptacles.add(info_n['fwd_cell'+str(i)].objectId) # add to the opened_receptacles list
                        self.closed_receptacles.discard(info_n['fwd_cell'+str(i)].objectId)
                        if info_n['fwd_cell'+str(i)].objectId in self.checked_receptacles: # if the receptacle was checked before, penalize
                            reward += -1.0 * self.open_reward_coeff
                        else:                                    # else, if it was not checked, give reward
                            self.checked_receptacles.add(info_n['fwd_cell'+str(i)].objectId)
                            reward += 1.0 * self.open_reward_coeff
                    
                    elif info_n['fwd_cell'+str(i)].objectId in self.opened_receptacles: # penalty for open opened_receptacles
                        reward += -1.0 * self.open_reward_coeff
                elif info_n['fwd_cell'+str(i)].can_toggle() and action[i] == self.Actions.close:            
                    if info_n['fwd_cell'+str(i)].objectId in self.opened_receptacles: # if the receptacle was opened
                        self.closed_receptacles.add(info_n['fwd_cell'+str(i)].objectId) # add to the closed_receptacles list
                        self.opened_receptacles.discard(info_n['fwd_cell'+str(i)].objectId)
                        
                    elif info_n['fwd_cell'+str(i)].objectId in self.closed_receptacles:
                        reward += -1.0 * self.open_reward_coeff # penalty for close closed_receptacles    
        
        
            # reward(penalize) for carrying (non)target object or 
            # just finish the episode
            if self.agents[i].carrying != None:
                if self.objIdx == self.agents[i].carrying.objectId:
                    self.planning_mode = self.PlanningMode.bring
                    if self.agents[i].preCarrying is None:
                        reward += 1.0 * self.carry_reward_coeff
                else:
                    if self.agents[i].preCarrying is None:
                        reward += -1.0 * self.carry_reward_coeff

            # If successfully dropping an object into the target
            u, v = self.dir_vec_i(i)
            ox, oy = (self.agents[i].pos[0] + u, self.agents[i].pos[1] + v)
            tx, ty = self.target_pos
            if action[i] == self.Actions.drop and self.agents[i].preCarrying:
                front_obj = self.grid.get(ox,oy)
                if front_obj.type is 'goal':
                    if abs(ox - tx) == 0 and abs(oy - ty) == 0:
                        done = True
                        self.success = True
                        #reward = 1.0
                        reward += 1.0
            
            
        # coverage reward
        reward += self.new_coverage * self.coverage_reward_coeff

        # reward_for_test
        # reward = 0

        
                    
        
        # if step num exceed 200, done
        if self.steps_remaining <= 0:
            reward += -0.1
            done = True
        

        self.new_coverage = 0
        #print("obs: ")
        #print(obs)
        
        #print(self.opened_receptacles)
        #print(self.closed_receptacles)
        #print(self.checked_receptacles)
        obs = self.generate_network_input_one_hot_for_mlp(obs)
        return obs, reward, done, info_n

    def publish_observation(self, image):
        ros_msg = Obs_multi()
        for k in range(len(image)):
            agent_observation = Obs()
            for i in range(self.agent_view_size):
                for j in range(self.agent_view_size):
    #                print("{}'th column, {}'th row".format(i,j))
                    abs_i, abs_j = self.get_world_coordinate_i(i, j, k)
    #                print("{}, {}".format(abs_i, abs_j))
                    if image[k][i][j][0] == OBJECT_TO_IDX['wall']:
                        agent_observation.type_id_list.append(image[k][i][j][0])
                        agent_observation.object_id_list.append(0)
                        agent_observation.object_pos_list.append(abs_i)
                        agent_observation.object_pos_list.append(abs_j)
                        agent_observation.object_state_list.append(0)
                    
                    elif image[k][i][j][0] == OBJECT_TO_IDX['box']:
                        agent_observation.type_id_list.append(image[k][i][j][0])
                        agent_observation.object_id_list.append(image[k][i][j][3])
                        agent_observation.object_pos_list.append(abs_i)
                        agent_observation.object_pos_list.append(abs_j)
                        agent_observation.object_state_list.append(image[k][i][j][2])
                        
                    elif image[k][i][j][0] == OBJECT_TO_IDX['empty']:
                        agent_observation.type_id_list.append(image[k][i][j][0])
                        agent_observation.object_id_list.append(0)
                        agent_observation.object_pos_list.append(abs_i)
                        agent_observation.object_pos_list.append(abs_j)
                        agent_observation.object_state_list.append(0)
                        
                    elif image[k][i][j][0] == OBJECT_TO_IDX['ball']:
                        agent_observation.type_id_list.append(image[k][i][j][0])
                        agent_observation.object_id_list.append(image[k][i][j][3])
                        agent_observation.object_pos_list.append(abs_i)
                        agent_observation.object_pos_list.append(abs_j)
                        agent_observation.object_state_list.append(image[k][i][j][2])
    
                    elif image[k][i][j][0] == OBJECT_TO_IDX['goal']:
                        agent_observation.type_id_list.append(image[k][i][j][0])
                        agent_observation.object_id_list.append(0)
                        agent_observation.object_pos_list.append(abs_i)
                        agent_observation.object_pos_list.append(abs_j)
                        agent_observation.object_state_list.append(0)

            agent_observation.agent_dir = self.agents[k].dir
            ros_msg.observation_list.append(agent_observation)
        self.observation_pub.publish(ros_msg)        
        
    def dispatch_plan(self, actions, render = False):
        print("dispatch plan called")
        reward_sum = 0
        obs, reward, done, info = None, None, None, {'fwd_cell': None}
        if self.dispatch_plan_action_id - self.prev_dispatch_plan_action_id > 0 and len(actions) > 0:
            precondition_check = self.check_precondition(actions[self.dispatch_plan_action_id][0])
            if precondition_check == False:
                print("dispatch_plan: {} precondition not achieved".format(actions[self.dispatch_plan_action_id][0].name))
                self.dispatch_plan_action_id = 0
                self.prev_dispatch_plan_action_id = -1
                return None, None, None, None
        # dispatch plan until potential goal update
        # 'actions[self.dispatch_plan_action_id][2] == False' means it is not potential goal update related semantic action
        print("actions length: {}, dispatch_plan_id: {}".format(len(actions), self.dispatch_plan_action_id))
        print("actions self.dispatch_plan_action_id length: {}".format(len(actions[self.dispatch_plan_action_id])))

        while actions[self.dispatch_plan_action_id][2] == False:
            print("actions length: {}, dispatch_plan_id: {}".format(len(actions), self.dispatch_plan_action_id))
            print("actions self.dispatch_plan_action_id length: {}".format(len(actions[self.dispatch_plan_action_id])))

            while len(actions[self.dispatch_plan_action_id][1]) > 0:    
                action = actions[self.dispatch_plan_action_id][1].pop(0)
                obs, reward, done, info = self.step(action)
                if render:
                    self.render()
                reward_sum += reward
            self.prev_dispatch_plan_action_id = self.dispatch_plan_action_id        
    
            # if actions for semantic action is done (= actions[dispatch_plan_action_id][1] is empty)
            if not actions[self.dispatch_plan_action_id][1]:
                self.process_action_effect(actions[self.dispatch_plan_action_id][0])
                if len(actions)-1 > self.dispatch_plan_action_id:
                    self.dispatch_plan_action_id += 1
                else:
                    break
        
        if self.dispatch_plan_action_id + 1 <= len(actions):
            while len(actions[self.dispatch_plan_action_id][1]) > 0:    
                action = actions[self.dispatch_plan_action_id][1].pop(0)
                obs, reward, done, info = self.step(action)
                if render:
                    self.render()
                reward_sum += reward
            self.prev_dispatch_plan_action_id = self.dispatch_plan_action_id        
        
            # if actions for semantic action is done (= actions[dispatch_plan_action_id][1] is empty)
            if not actions[self.dispatch_plan_action_id][1]:
                self.process_action_effect(actions[self.dispatch_plan_action_id][0])
                if len(actions)-1 > self.dispatch_plan_action_id:
                    self.dispatch_plan_action_id += 1
        
        return obs, reward_sum, done, info        
        
    def invoke(self, meta_action, render = False):
        reward_sum = 0
        assert len(meta_action) == self.num_agents
        #print("mode: {}, meta_action: {}".format(self.mode, meta_action))
        done = False
        info = {'fwd_cell': None}
        explore_agent_id_list = []
        plan_agent_id_list = []
        scan_agent_id_list = []
        for i in range(len(meta_action)):
            if meta_action[i] == self.meta_actions.explore:
                explore_agent_id_list.append(i)
                self.agents[i].mode = self.Option_mode.explore
                
            elif meta_action[i] == self.meta_actions.plan:
                plan_agent_id_list.append(i)
                self.agents[i].mode = self.Option_mode.plan
                
            elif meta_action[i] == self.meta_actions.scan:
                scan_agent_id_list.append(i)
                self.agents[i].mode = self.Option_mode.scan    
                
        if len(explore_agent_id_list) > 0:
            self.explore(explore_agent_id_list)
        
        if len(plan_agent_id_list) > 0:
            self.plan(plan_agent_id_list)

        if len(scan_agent_id_list) > 0:
            self.scan(scan_agent_id_list)
        
        min_option_len = 1000  # 1000 is nothing special, just for simple infinity value
        for i in range(len(self.agents)):
            if len(self.agents[i].action_list) < min_option_len:
                min_option_len = len(self.agents[i].action_list)
        
        if self.meaningless:            
            done = True
            info['is_mission_succeeded'] = self.success
            info['mission_completion_time'] = self.max_steps - self.steps_remaining
            obs = self.gen_obs()
            map = self.generate_network_input_one_hot_for_mlp(obs)
            return map, -0.1, done, info
        
             
        for i in range(min_option_len):
            action = []
            for j in range(self.num_agents):
                action.append(self.agents[j].action_list.pop(0))
            obs, reward, done, info = self.step(action)
            
            
            reward_sum += reward
            if done:
                info['is_mission_succeeded'] = self.success
                info['mission_completion_time'] = self.max_steps - self.steps_remaining
                return obs, reward_sum, done, info
            
        info['is_mission_succeeded'] = self.success
        info['mission_completion_time'] = self.max_steps - self.steps_remaining
        
        
        obs = self.gen_obs()
        map = self.generate_network_input_one_hot_for_mlp(obs)
        #print("map: ")
        #print(map)
        print('%s, Overall reward=%.2f' % (meta_action, reward_sum))
        #print("obs: {}, reard_sum: {}, {}, {}, {}".format(type(obs), type(reward_sum), type(done), type(info), type(map)))
        #return obs, reward_sum, done, info
        return map, reward_sum, done, info #, map
        
        #print('%s, Overall reward=%.2f' % (meta_action, reward_sum))
        #return obs, reward_sum, done, info
             

        #obs = self.gen_obs()
        # map = self.generate_network_input(obs)
        #map = self.generate_network_input_one_hot_for_mlp(obs)
        #print("map: ")
        #print(map)
        #print('%s, Overall reward=%.2f' % (meta_action, reward_sum))
        #print("{}, {}, {}, {}, {}".format(type(obs), type(reward_sum), type(done), type(info), type(map)))
        #return obs, reward_sum, done, info
        #return map, reward_sum, done, info #, map

    def generate_network_input_one_hot_for_mlp(self, obs):
        temp_ball_map = copy.deepcopy(self.ball_map_one_hot)
        map = np.zeros(shape=(20, 20, 10), dtype=np.uint8)
        
        for i in range(self.num_agents):
            obs_grid, _ = Grid.decode(obs['image'][i])
            object = obs_grid.get(obs_grid.width//2, obs_grid.height-1)
            wx, wy = self.get_world_coordinate_i(obs_grid.width//2, obs_grid.height-1, i)
            map[self.agents[i].pos[0],self.spatial_map.map.info.height-self.agents[i].pos[1],7+i] = 1
            if np.array_equal(self.agents[i].pos, np.array([wx,wy])):
                wy = self.spatial_map.map.info.height - wy - 1
                #self.spatial_map.update_cell(np.array([wx,wy]), SpatialMap.OccGridStates.free
                if object is not None and object.type == 'ball':
                    temp_ball_map.update_cell(np.array([wx, wy]), 255)
        map[:,:,0] = copy.deepcopy(np.reshape(temp_ball_map.map.data, (self.ball_map_one_hot.map.info.width, self.ball_map_one_hot.map.info.height)))        
        map[:,:,1] = copy.deepcopy(np.reshape(self.box_map_one_hot.map.data, (self.ball_map_one_hot.map.info.width, self.ball_map_one_hot.map.info.height)))
        map[:,:,2] = copy.deepcopy(np.reshape(self.checked_box_map_one_hot.map.data, (self.ball_map_one_hot.map.info.width, self.ball_map_one_hot.map.info.height)))
        map[:,:,3] = copy.deepcopy(np.reshape(self.floor_map_one_hot.map.data, (self.ball_map_one_hot.map.info.width, self.ball_map_one_hot.map.info.height)))
        map[:,:,4] = copy.deepcopy(np.reshape(self.goal_map_one_hot.map.data, (self.ball_map_one_hot.map.info.width, self.ball_map_one_hot.map.info.height)))
        map[:,:,5] = copy.deepcopy(np.reshape(self.unknown_map_one_hot.map.data, (self.ball_map_one_hot.map.info.width, self.ball_map_one_hot.map.info.height)))
        map[:,:,6] = copy.deepcopy(np.reshape(self.wall_map_one_hot.map.data, (self.ball_map_one_hot.map.info.width, self.ball_map_one_hot.map.info.height)))
        flatten_map = np.reshape(map, (20*20*10,))
        
        for i in range(self.num_agents):
            flatten_map = np.append(flatten_map, self.agents[i].carrying is not None)
        #print("gen_net_input_one_hot_for_mlp's return shape: {}".format(flatten_map.shape))
        return flatten_map


    def generate_network_input_one_hot(self, obs):
        temp_ball_map = copy.deepcopy(self.ball_map_one_hot)
        map = np.zeros(shape=(50, 50, 7), dtype=np.uint8)

        obs_grid, _ = Grid.decode(obs['image'])
        object = obs_grid.get(obs_grid.width//2, obs_grid.height-1)
        wx, wy = self.get_world_coordinate(obs_grid.width//2, obs_grid.height-1)
        if np.array_equal(self.agent_pos, np.array([wx,wy])):
            wy = self.spatial_map.map.info.height - wy - 1
            #self.spatial_map.update_cell(np.array([wx,wy]), SpatialMap.OccGridStates.free
            if object is not None and object.type == 'ball':
                temp_ball_map.update_cell(np.array([wx, wy]), 1)
        map[:,:,0] = copy.deepcopy(np.reshape(temp_ball_map.map.data, (self.ball_map_one_hot.map.info.width, self.ball_map_one_hot.map.info.height)))        
        map[:,:,1] = copy.deepcopy(np.reshape(self.box_map_one_hot.map.data, (self.ball_map_one_hot.map.info.width, self.ball_map_one_hot.map.info.height)))
        map[:,:,2] = copy.deepcopy(np.reshape(self.checked_box_map_one_hot.map.data, (self.ball_map_one_hot.map.info.width, self.ball_map_one_hot.map.info.height)))
        map[:,:,3] = copy.deepcopy(np.reshape(self.floor_map_one_hot.map.data, (self.ball_map_one_hot.map.info.width, self.ball_map_one_hot.map.info.height)))
        map[:,:,4] = copy.deepcopy(np.reshape(self.goal_map_one_hot.map.data, (self.ball_map_one_hot.map.info.width, self.ball_map_one_hot.map.info.height)))
        map[:,:,5] = copy.deepcopy(np.reshape(self.unknown_map_one_hot.map.data, (self.ball_map_one_hot.map.info.width, self.ball_map_one_hot.map.info.height)))
        map[:,:,6] = copy.deepcopy(np.reshape(self.wall_map_one_hot.map.data, (self.ball_map_one_hot.map.info.width, self.ball_map_one_hot.map.info.height)))
        return map
        
    def generate_network_input(self, obs):
        map = copy.deepcopy(self.object_map_mlp)
        #map = np.reshape(self.object_map.map.data, (self.width*self.object_map.factor, self.height*self.object_map.factor))
        obs_grid, _ = Grid.decode(obs['image'])
        object = obs_grid.get(obs_grid.width//2, obs_grid.height-1)
        wx, wy = self.get_world_coordinate(obs_grid.width//2, obs_grid.height-1)
        if np.array_equal(self.agent_pos, np.array([wx,wy])):
            wy = self.spatial_map.map.info.height - wy - 1
            #self.spatial_map.update_cell(np.array([wx,wy]), SpatialMap.OccGridStates.free
            if object is not None and object.type == 'ball':
                map.update_cell(np.array([wx, wy]), ObjectMap_MLP.ObjGridStates.ball+ObjectMap_MLP.ObjGridStates.agent)
            else:
                value = map.get_value(np.array([wx, wy]))
                map.update_cell(np.array([wx, wy]), value + ObjectMap_MLP.ObjGridStates.agent)
        flatten_map = np.reshape(map, (self.width*self.height,))
        return flatten_map
    
    def get_view_exts_i(self, id):
        """
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set
        """

        # Facing right
        if self.agents[id].dir == 0:
            topX = self.agents[id].pos[0]
            topY = self.agents[id].pos[1] - self.agent_view_size // 2
        # Facing down
        elif self.agents[id].dir == 1:
            topX = self.agents[id].pos[0] - self.agent_view_size // 2
            topY = self.agents[id].pos[1]
        # Facing left
        elif self.agents[id].dir == 2:
            topX = self.agents[id].pos[0] - self.agent_view_size + 1
            topY = self.agents[id].pos[1] - self.agent_view_size // 2
        # Facing up
        elif self.agents[id].dir == 3:
            topX = self.agents[id].pos[0] - self.agent_view_size // 2
            topY = self.agents[id].pos[1] - self.agent_view_size + 1
        else:
            assert False, "invalid agent direction"

        botX = topX + self.agent_view_size
        botY = topY + self.agent_view_size

        return (topX, topY, botX, botY)

    def gen_obs_grid_i(self, id):
        """
        Generate the sub-grid observed by the agent[id].
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        """

        topX, topY, botX, botY = self.get_view_exts_i(id)

        grid = self.grid.slice(topX, topY, self.agent_view_size, self.agent_view_size)

        for i in range(self.agents[id].dir + 1):
            grid = grid.rotate_left()

        # Process occluders and visibility
        # Note that this incurs some performance cost
        if not self.see_through_walls:
            vis_mask = grid.process_vis(agent_pos=(self.agent_view_size // 2 , self.agent_view_size - 1))
        else:
            vis_mask = np.ones(shape=(grid.width, grid.height), dtype=np.bool)

        # Make it so the agent sees what it's carrying
        # We do this by placing the carried object at the agent's position
        # in the agent's partially observable view
        agent_pos = grid.width // 2, grid.height - 1
        if self.agents[id].carrying:
            grid.set(*agent_pos, self.agents[id].carrying)
        else:
            grid.set(*agent_pos, None)

        return grid, vis_mask
    
    def gen_obs(self):
        grid = [None] * self.num_agents
        vis_mask = [None] * self.num_agents
        image = [None] * self.num_agents
        for id in range(self.num_agents):
            grid[id], vis_mask[id] = self.gen_obs_grid_i(id)
            
            # Encode the partially observable view into a numpy array
            image[id] = grid[id].encode(vis_mask[id])
            assert hasattr(self, 'mission'), "environments must define a textual mission string"
            
        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - a textual mission string (instructions for the agent)
        obs = {
            'image': image,
            'mission': self.mission
        }
        return obs
    
    def get_world_coordinate_i(self, vx, vy, id):
        ax, ay = self.agents[id].pos
        dx, dy = self.dir_vec_i(id)
        rx, ry = self.right_vec_i(id)
        
        sz = self.agent_view_size
        hs = self.agent_view_size // 2
        tx = ax + (dx * (sz-1)) - (rx * hs)
        ty = ay + (dy * (sz-1)) - (ry * hs)
        
        lx = (-dy*vx - ry*vy) // (-rx*dy + ry*dx)
        ly = (dx*vx + rx*vy) // (-rx*dy + ry*dx)
        
        wx = lx + tx
        wy = ly + ty
        return wx, wy    
        
    def render(self, mode = 'human', close = False, highlight = True, tile_size = TILE_PIXELS):
        if close:
            if self.window:
                self.window.close()
            return

        if mode == 'human' and not self.window:
            import gym_minigrid.window
            self.window = gym_minigrid.window.Window('gym_minigrid')
            self.window.show(block=False)

        # Mask of which cells to highlight
        highlight_mask = np.zeros(shape=(self.width, self.height), dtype=np.bool)
 
        for i in range(len(self.agents)):
            # Compute which cells are visible to the agent
            _, vis_mask = self.gen_obs_grid_i(i)
    
            # Compute the world coordinates of the bottom-left corner
            # of the agent's view area
            #f_vec = self.dir_vec
            #r_vec = self.right_vec
            #top_left = self.agent_pos + f_vec * (self.agent_view_size-1) - r_vec * (self.agent_view_size // 2)
    
    
            # For each cell in the visibility mask
            for vis_j in range(0, self.agent_view_size):
                for vis_i in range(0, self.agent_view_size):
                    # If this cell is not visible, don't highlight it
                    if not vis_mask[vis_i, vis_j]:
                        continue
    
                    # Compute the world coordinates of this cell
                    abs_i, abs_j = self.get_world_coordinate_i(vis_i,vis_j, i)
    
                    if abs_i < 0 or abs_i >= self.width:
                        continue
                    if abs_j < 0 or abs_j >= self.height:
                        continue
    
                    # Mark this cell to be highlighted
                    highlight_mask[abs_i, abs_j] = True

        agents_pos = []
        agents_dir = []
        for agent in self.agents:
            agents_pos.append(agent.pos)
            agents_dir.append(agent.dir)
    
        # Render the whole grid
        img = self.grid.render_multi(
            tile_size,
            agents_pos,
            agents_dir,
            highlight_mask=highlight_mask if highlight else None
        )

        if mode == 'human':
            self.window.set_caption(self.mission)
            self.window.show_img(img)

        obs = self.gen_obs()
        if self.render_counter == 0:
            self.update_maps(obs, None)
            self.spatial_map_pub.publish(self.spatial_map.map)
            self.object_map_pub.publish(self.object_map.map)
            self.publish_ros_agent_pos()
#            self.agent_init_dir = self.agent_dir
#            self.agent_init_pos_pub.publish(self.agent_init_pos)
            self.publish_observation(obs['image'])
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
        ros_agent_pos = PoseArray()
        ros_agent_pos.header.frame_id = "map"
        ros_agent_pos.header.stamp = rospy.Time.now()
        #print("self.agent_pos: %d, %d" % (self.agent_pos[0], self.agent_pos[1]) )
        for i in range(len(self.agents)):
            pose = Pose()
            pose.position.x = (self.agents[i].pos[0]+0.5) * self.spatial_map.map.info.resolution 
            pose.position.y = (self.spatial_map.map.info.height - self.agents[i].pos[1] - 1 + 0.5) * self.spatial_map.map.info.resolution
            pose.position.z = 0
            orientation = self.dir_to_quaternion(i)
            pose.orientation.x = orientation[0]
            pose.orientation.y = orientation[1]
            pose.orientation.z = orientation[2]
            pose.orientation.w = orientation[3]
            
            pose_vis = PoseStamped()
            pose_vis.header.frame_id = "map"
            pose_vis.header.stamp = rospy.Time.now()
            pose_vis.pose.position.x = pose.position.x
            pose_vis.pose.position.y = pose.position.y
            pose_vis.pose.position.z = pose.position.z
            pose_vis.pose.orientation.x = pose.orientation.x
            pose_vis.pose.orientation.y = pose.orientation.y
            pose_vis.pose.orientation.z = pose.orientation.z
            pose_vis.pose.orientation.w = pose.orientation.w
            self.agents_pose_pub_vis[i].publish(pose_vis)
            ros_agent_pos.poses.append(pose)
        self.agent_pos_pub.publish(ros_agent_pos)
        return ros_agent_pos

    def publish_ros_goal_pos(self, pos):
        goal_pos = PoseStamped()
        goal_pos.header.frame_id = "map"
        goal_pos.header.stamp = rospy.Time.now()
        goal_pos.pose.position.x = pos[0]
        goal_pos.pose.position.y = pos[1]
        self.goal_pos_pub.publish(goal_pos)
        return goal_pos

    def dir_to_quaternion(self, id):
        dir = self.agents[id].dir
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
        #print(obs['image'])
        #print(self.agent_dir)
        for i in range(len(self.agents)):
            #print("i: {}".format(i))
            if np.array_equal(self.agents[i].prev_pos, self.agents[i].pos) and np.array_equal(self.agents[i].prev_dir, self.agents[i].dir): # no movement
                if action is not None and (action[i] == self.Actions.open or action[i] == self.Actions.pickup or action[i] == self.Actions.drop):
                    front = self.agents[i].front_pos()
                    fwd_cell = self.grid.get(front[0], front[1])
    
                    front[1] = self.spatial_map.map.info.height - front[1] - 1
                    if fwd_cell is None: # empty cell or out of range
                        if 0 <= front[0] and front[0] <= self.spatial_map.map.info.width-1 and 0 <= front[1] and front[1] <= self.spatial_map.map.info.height-1:
                            self.spatial_map.update_cell(front, SpatialMap.OccGridStates.free)
                            self.object_map.update_cell(front, ObjectMap.ObjGridStates.floor)
                            self.floor_map_one_hot.update_cell(front, 255)
                            self.unknown_map_one_hot.update_cell(front, 0)
                            self.object_map_mlp.update_cell(front, ObjectMap_MLP.ObjGridStates.floor)
                    else:
                        
                        # update object map
                        if fwd_cell.type == 'box':
                            self.seen_receptacles.add(fwd_cell.objectId)
                            #print("fwd_cell.objectId: %d" % fwd_cell.objectId)
                            if fwd_cell.isOpen == True:
                                if fwd_cell.contains is not None:
                                    if fwd_cell.contains.type == 'ball':
                                        self.planning_mode = self.PlanningMode.keep
                                        self.object_map.update_cell(front, ObjectMap.ObjGridStates.ball, center_only = True)
                                        self.ball_map_one_hot.update_cell(front, 255)
                                        self.unknown_map_one_hot.update_cell(front, 0)
                                        self.object_map_mlp.update_cell(front, ObjectMap_MLP.ObjGridStates.ball + ObjectMap_MLP.ObjGridStates.box)
    
                                    elif fwd_cell.contains.type == 'key':
                                        self.object_map.update_cell(front, ObjectMap.ObjGridStates.key, center_only = True)
                                else:
                                    self.object_map.update_cell(front, ObjectMap.ObjGridStates.checked_box)
                                    self.checked_box_map_one_hot.update_cell(front, 255)
                                    self.unknown_map_one_hot.update_cell(front, 0)
                                    self.object_map_mlp.update_cell(front, ObjectMap_MLP.ObjGridStates.checked_box)
    
                            elif fwd_cell.objectId not in self.checked_receptacles:
                                self.object_map.update_cell(front, ObjectMap.ObjGridStates.box)
                                self.box_map_one_hot.update_cell(front, 255)
                                self.unknown_map_one_hot.update_cell(front, 0)
                                self.object_map_mlp.update_cell(front, ObjectMap_MLP.ObjGridStates.box)
    
                        elif fwd_cell.type == 'key':
                            self.object_map.update_cell(front, ObjectMap.ObjGridStates.key, center_only = True)
                            self.unknown_map_one_hot.update_cell(front, 0)
                            
                        elif fwd_cell.type == 'ball':
                            self.object_map.update_cell(front, ObjectMap.ObjGridStates.ball, center_only = True)
                            self.ball_map_one_hot.update_cell(front, 255)
                            self.unknown_map_one_hot.update_cell(front, 0)
                            self.object_map_mlp.update_cell(front, ObjectMap_MLP.ObjGridStates.ball)
                            
                        elif fwd_cell.type == 'wall':
                            self.object_map.update_cell(front, ObjectMap.ObjGridStates.wall)
                            self.wall_map_one_hot.update_cell(front, 255)
                            self.unknown_map_one_hot.update_cell(front, 0)
                            self.object_map_mlp.update_cell(front, ObjectMap_MLP.ObjGridStates.wall)
                            
                        elif fwd_cell.type == 'goal':
                            self.object_map.update_cell(front, ObjectMap.ObjGridStates.goal)
                            self.object_map_mlp.update_cell(front, ObjectMap_MLP.ObjGridStates.goal)
                            self.goal_map_one_hot.update_cell(front, 255)
                            self.unknown_map_one_hot.update_cell(front, 0)
                        else:
                            print("update_maps: this should not happen. new type")
                    
                        # update spatial map
                        if fwd_cell.can_overlap():
                            self.spatial_map.update_cell(front, SpatialMap.OccGridStates.free)
                        else:
                            self.spatial_map.update_cell(front, SpatialMap.OccGridStates.occupied)
        
                            
                else:
                    continue
            else:
                # for all the cells in the agent's view, update the cells info into the map unless it is not chekced receptacles
                obs_grid, _ = Grid.decode(obs['image'][i])
                for k in range(obs_grid.width):
                    for j in range(obs_grid.height):
                        object = obs_grid.get(k,j)
                        wx, wy = self.get_world_coordinate_i(k,j, i)
                        wy = self.spatial_map.map.info.height - wy - 1
                        agent_pos = np.array([self.agents[i].pos[0], self.spatial_map.map.info.height - self.agents[i].pos[1] - 1])
                        #print("i, wx, wy: {}, {}, {}, agent_pos: {}, {}".format(i, wx, wy, self.agents[i].pos[0], self.agents[i].pos[1]))
                        if np.array_equal(agent_pos, np.array([wx,wy])):
                            #print("i, wx, wy: {}, {}, {}".format(i, wx, wy))                          
                            self.spatial_map.update_cell(np.array([wx,wy]), SpatialMap.OccGridStates.free)
                            continue
    
                        if object is None: # empty cell or out of range
                            if 0 <= wx and wx <= self.spatial_map.map.info.width-1 and 0 <= wy and wy <= self.spatial_map.map.info.height-1:
                                self.spatial_map.update_cell(np.array([wx,wy]), SpatialMap.OccGridStates.free)
                                self.object_map.update_cell(np.array([wx,wy]), ObjectMap.ObjGridStates.floor)
                                self.floor_map_one_hot.update_cell(np.array([wx,wy]), 255)
                                self.unknown_map_one_hot.update_cell(np.array([wx,wy]), 0)
                                self.object_map_mlp.update_cell(np.array([wx,wy]), ObjectMap_MLP.ObjGridStates.floor)
                        else:
                            # update object map
                            if object.type == 'box':
                                self.seen_receptacles.add(object.objectId)
                                if object.objectId in self.checked_receptacles:
                                    pass
                                
                                elif object.isOpen == True:
                                    self.opened_receptacles.add(object.objectId)
                                    if object.contains is not None:
                                        if object.contains.type == 'ball':
                                            self.object_map.update_cell(np.array([wx,wy]), ObjectMap.ObjGridStates.ball, center_only = True)
                                            self.ball_map_one_hot.update_cell(np.array([wx,wy]), 255)
                                            self.unknown_map_one_hot.update_cell(np.array([wx,wy]), 0)
                                            self.object_map_mlp.update_cell(np.array([wx,wy]), ObjectMap_MLP.ObjGridStates.ball + ObjectMap_MLP.ObjGridStates.box)
    
                                        elif object.contains.type == 'key':
                                            self.object_map.update_cell(np.array([wx,wy]), ObjectMap.ObjGridStates.key, center_only = True)
                                    else:
                                        self.object_map.update_cell(np.array([wx,wy]), ObjectMap.ObjGridStates.checked_box)
                                        self.box_map_one_hot.update_cell(np.array([wx,wy]), 255)
                                        self.unknown_map_one_hot.update_cell(np.array([wx,wy]), 0)
                                        self.object_map_mlp.update_cell(np.array([wx,wy]), ObjectMap_MLP.ObjGridStates.checked_box)
                                        self.checked_receptacles.add(object.objectId)
    
                                else:
                                    self.closed_receptacles.add(object.objectId)
                                    self.object_map.update_cell(np.array([wx, wy]), ObjectMap.ObjGridStates.box)
                                    self.box_map_one_hot.update_cell(np.array([wx,wy]), 255)
                                    self.unknown_map_one_hot.update_cell(np.array([wx,wy]), 0)
                                    self.object_map_mlp.update_cell(np.array([wx,wy]), ObjectMap_MLP.ObjGridStates.box)
                                    
                            elif object.type == 'key':
                                self.object_map.update_cell(np.array([wx,wy]), ObjectMap.ObjGridStates.key, center_only = True)
                                
                            elif object.type == 'ball':
                                self.object_map.update_cell(np.array([wx,wy]), ObjectMap.ObjGridStates.ball, center_only = True)
                                self.ball_map_one_hot.update_cell(np.array([wx,wy]), 255)
                                self.unknown_map_one_hot.update_cell(np.array([wx,wy]), 0)
                                self.object_map_mlp.update_cell(np.array([wx,wy]), ObjectMap_MLP.ObjGridStates.ball)
    
                            elif object.type == 'wall':
                                self.object_map.update_cell(np.array([wx, wy]), ObjectMap.ObjGridStates.wall)
                                self.wall_map_one_hot.update_cell(np.array([wx,wy]), 255)
                                self.unknown_map_one_hot.update_cell(np.array([wx,wy]), 0)
                                self.object_map_mlp.update_cell(np.array([wx,wy]), ObjectMap_MLP.ObjGridStates.wall)
    
                            elif object.type == 'goal':
                                self.object_map.update_cell(np.array([wx, wy]), ObjectMap.ObjGridStates.goal)
                                self.goal_map_one_hot.update_cell(np.array([wx,wy]), 255)
                                self.unknown_map_one_hot.update_cell(np.array([wx,wy]), 0)
                                self.object_map_mlp.update_cell(np.array([wx,wy]), ObjectMap_MLP.ObjGridStates.goal)
    
                            else:
                                print("update_maps: this should not happen. new type")
                                
                            
                            index = self.spatial_map.xy_to_index(wx, wy)
                            if self.spatial_map.map.data[index] == SpatialMap.OccGridStates.unknown:
                                self.new_coverage += 1
    
                            # update spatial map
                            if object.can_overlap():
                                self.spatial_map.update_cell(np.array([wx,wy]), SpatialMap.OccGridStates.free)
                            else:
                                self.spatial_map.update_cell(np.array([wx,wy]), SpatialMap.OccGridStates.occupied)

                
        self.spatial_map.map.header.stamp = rospy.Time.now()
        self.object_map.map.header.stamp = rospy.Time.now()                   

    def scan(self, id_list):
        for i in id_list:
            self.agents[i].action_list = [self.Actions.left] * 4

    def explore(self, id_list):
        #print("multi_hiprlgrid: explore called")
        rospy.wait_for_service('init_pose_list_update' + str(self.process_num))
        result = False
        try:
            init_pose_update = rospy.ServiceProxy('init_pose_list_update' + str(self.process_num), InitPosList)
            result = init_pose_update([self.agent_init_pos.poses[i].position.x for i in range(self.num_agents)],
                                      [self.agent_init_pos.poses[i].position.y for i in range(self.num_agents)], 
                                      [self.agent_init_dir[i] for i in range(self.num_agents)],
                                      id_list)
            
        except rospy.ServiceException as e:
            print("Service call failed: %s" %e)
        #print("init_pose_list_update: ", result)

        try:
            pose_update = rospy.ServiceProxy('pose_list_update' + str(self.process_num), InitPosList)
            x = [(self.agents[i].pos[0] + 0.5) * self.spatial_map.map.info.resolution for i in range(self.num_agents)]
            y = [(self.spatial_map.map.info.height - self.agents[i].pos[1] - 0.5) * self.spatial_map.map.info.resolution for i in range(self.num_agents)]
            dir = [self.agents[i].dir for i in range(self.num_agents)]
            result = pose_update(x, y, dir, id_list)
        except rospy.ServiceException as e:
            print("Service call failed: %s" %e)
        
        #print("pose_list_update: ", result)       
 
        # add agents locations as obstacle to copied map instance which is for navigation
        navigation_map = copy.deepcopy(self.spatial_map)
        for i in range(self.num_agents):
            agent_x = self.agents[i].pos[0]
            agent_y = self.spatial_map.map.info.height - self.agents[i].pos[1] - 1
            navigation_map.update_cell([agent_x, agent_y],self.spatial_map.OccGridStates.occupied)

        # Execute mapConvert in findFrontier, which generates plan
        self.navigation_map_pub.publish(navigation_map.map)
        
        counter = 0
        while(self.explore_action_set is False):
            #print("explore_action_set: ", self.explore_action_set)
            counter = counter + 1
            time.sleep(0.1)
            if counter > 10:
                return
            #block until explore plan is subscribed and updated
            
        fail_counter = 0
        for i in id_list:
            if self.agents[i].mode != self.Option_mode.explore:
                continue
            #print("agent {} plan: {}".format(i, self.agents[i].action_list))
            if len(self.agents[i].action_list) == 0 or self.agents[i].action_list[0] == self.Actions.done:
                fail_counter = fail_counter + 1
                
#        if fail_counter == len(id_list):
#            for i in range(len(id_list)):
#                self.navigation_map_pub.publish(self.agents[i].map)
        self.explore_action_set = False
        
        
        
        # step using actions extracted from explore_action_list
        # temporally, set left
        #self.explore_action_list = [self.Actions.left] * 4
        return
    
    def plan(self, id_list):
        #print("plan in planning_mode: {}".format(self.planning_mode))
        try:
            pose_update = rospy.ServiceProxy('pose_list_update' + str(self.process_num), InitPosList)
            x = [(self.agents[i].pos[0] + 0.5) * self.spatial_map.map.info.resolution for i in range(self.num_agents)]
            y = [(self.spatial_map.map.info.height - self.agents[i].pos[1] - 0.5) * self.spatial_map.map.info.resolution for i in range(self.num_agents)]
            dir = [self.agents[i].dir for i in range(self.num_agents)]
            result = pose_update(x, y, dir, id_list)
        except rospy.ServiceException as e:
            result = False
            print("Service call failed: %s" %e)
        
        #print("pose_list_update: ", result)       

        
        self.navigation_map_pub.publish(self.spatial_map.map)
        
        if self.planning_mode == self.PlanningMode.search:
            #print("Allocating tasks / generate plan for planning agents to search")
            rospy.wait_for_service('/task_allocate'+ str(self.process_num))
            try:
                task_allocate = rospy.ServiceProxy('/task_allocate' + str(self.process_num), PlanningID)
                x = [(self.agents[i].pos[0] + 0.5) * self.spatial_map.map.info.resolution for i in id_list]
                y = [(self.spatial_map.map.info.height - self.agents[i].pos[1] - 0.5) * self.spatial_map.map.info.resolution for i in id_list]
                dir = [self.agents[i].dir for i in id_list]
                boxes_id = self.seen_receptacles - self.checked_receptacles
                #boxes_idx = np.nonzero(np.array(self.box_map_one_hot.map.data))
                #box_pos_x_list = []
                #box_pos_y_list = []    
                box_pos_x = []
                box_pos_y = []
    
                #box_pos_x, box_pos_y = self.box_map_one_hot.index_to_xy(boxes_idx[0])
                #print("boxes_id: {}".format(boxes_id))
                #print("self.boxPos: {}".format(self.boxPos))
                box_pos_list = [self.boxPos[i] for i in boxes_id]
                
                #print("box_pos_list: {}".format(box_pos_list))
                for box_pos in box_pos_list:
                    box_pos_x.append((box_pos[0]+0.5) * self.spatial_map.map.info.resolution)
                    box_pos_y.append((self.spatial_map.map.info.height - box_pos[1] - 0.5) * self.spatial_map.map.info.resolution)
                    
    
                resp = task_allocate(id_list, x, y, dir, boxes_id, box_pos_x, box_pos_y, self.planning_mode)
            except rospy.ServiceException as e:
                print("Planning Agent Update Service call failed: %s" %e)
                return None
        
        elif self.planning_mode == self.PlanningMode.keep:
            #print("Generate plan for planning agents to keep")
            rospy.wait_for_service('/task_allocate'+ str(self.process_num))
            try:
                task_allocate = rospy.ServiceProxy('/task_allocate' + str(self.process_num), PlanningID)
                x = [(self.agents[i].pos[0] + 0.5) * self.spatial_map.map.info.resolution for i in id_list]
                y = [(self.spatial_map.map.info.height - self.agents[i].pos[1] - 0.5) * self.spatial_map.map.info.resolution for i in id_list]
                dir = [self.agents[i].dir for i in id_list]
                boxes_id = [0]
                  
                box_pos_x = []
                box_pos_y = []
    
                #box_pos_x, box_pos_y = self.box_map_one_hot.index_to_xy(boxes_idx[0])
                #print("boxes_id: {}".format(boxes_id))
                #print("self.boxPos: {}".format(self.boxPos))
                box_pos_list = [self.boxPos[i] for i in boxes_id]
                #print("box_pos_list: {}".format(box_pos_list))
                for box_pos in box_pos_list:
                    box_pos_x.append((box_pos[0]+0.5) * self.spatial_map.map.info.resolution)
                    box_pos_y.append((self.spatial_map.map.info.height - box_pos[1] - 0.5) * self.spatial_map.map.info.resolution)
                    
    
                resp = task_allocate(id_list, x, y, dir, boxes_id, box_pos_x, box_pos_y, self.planning_mode)
            except rospy.ServiceException as e:
                print("Planning Agent Update Service call failed: %s" %e)
                return None
        
        elif self.planning_mode == self.PlanningMode.bring:
            #print("Generate plan for planning agents to bring")
            rospy.wait_for_service('/task_allocate'+ str(self.process_num))
            try:
                task_allocate = rospy.ServiceProxy('/task_allocate' + str(self.process_num), PlanningID)
                x = [(self.agents[i].pos[0] + 0.5) * self.spatial_map.map.info.resolution for i in id_list]
                y = [(self.spatial_map.map.info.height - self.agents[i].pos[1] - 0.5) * self.spatial_map.map.info.resolution for i in id_list]
                dir = [self.agents[i].dir for i in id_list]
                boxes_id = [0]
                
                box_pos_x = []
                box_pos_y = []
    
                #box_pos_x, box_pos_y = self.box_map_one_hot.index_to_xy(boxes_idx[0])
                #print("boxes_id: {}".format(boxes_id))
                #print("self.boxPos: {}".format(self.boxPos))
                box_pos_x.append((self.goal_pos[0]+0.5) * self.spatial_map.map.info.resolution)
                box_pos_y.append((self.spatial_map.map.info.height - self.goal_pos[1] - 0.5) * self.spatial_map.map.info.resolution)
                    
    
                resp = task_allocate(id_list, x, y, dir, boxes_id, box_pos_x, box_pos_y, self.planning_mode)
            except rospy.ServiceException as e:
                print("Planning Agent Update Service call failed: %s" %e)
                return None
        
        
        j = 0
        min_len = 1000
        for i in range(self.num_agents):
            if self.agents[i].mode != self.Option_mode.plan:
                continue

            if j >= len(resp.action_array.list):
                break

            if len(resp.action_array.list) > 0 and len(resp.action_array.list[j].data) < min_len and len(resp.action_array.list[j].data) > 0 and resp.action_array.list[j].data[0] >= 0:
                min_len = len(resp.action_array.list[j].data) 
            #print("agent {}'s plan action length: {}".format(i, len(resp.action_array.list[j].data)))
            self.agents[i].action_list  = [0] * len(resp.action_array.list[j].data)
            for k in range(len(resp.action_array.list[j].data)):
                self.agents[i].action_list[k] = resp.action_array.list[j].data[k]
                
            
            j = j + 1
        
        j = 0
        for i in range(self.num_agents):
            if self.agents[i].mode != self.Option_mode.plan:
                continue
            
            if j >= len(resp.action_array.list):
                break
            
            if len(resp.action_array.list) > 0 and len(resp.action_array.list[j].data) == 1 and resp.action_array.list[j].data[0] < 0:
                #self.agents[i].action_list  = [self.Actions.done] * min_len
                self.meaningless = True
            j = j + 1
        
        # in case all failed to plan
        if min_len == 1000:
            for i in range(self.num_agents):
                if self.agents[i].mode != self.Option_mode.plan:
                    continue
                #self.agents[i].action_list = [self.Actions.done]
                self.meaningless = True
        
        
        #self.generate_actions_from_complete_plan(id_list)
        #self.dispatch_plan_action_id = 0
        self.prev_dispatch_plan_action_id = -1
        self.complete_plan_flag = False
        return 
        

    
    def check_precondition(self, item):
        rospy.wait_for_service('/check_precondition' + str(self.process_num))
        try:
            check_precondition = rospy.ServiceProxy('/check_precondition' + str(self.process_num), ActionExecution)
            req = ActionExecutionRequest()
            req.name = item.name
            req.action_id = item.action_id
            req.parameters = copy.deepcopy(item.parameters)
            resp = check_precondition(req)
            return resp.success
        except rospy.ServiceException as e:
            print("Check Precondition Service call failed: %s" %e)
            return False
    
    def process_action_effect(self, item):
        rospy.wait_for_service('/process_action_effect' + str(self.process_num))
        try:
            check_precondition = rospy.ServiceProxy('/process_action_effect' + str(self.process_num), ActionExecution)
            req = ActionExecutionRequest()
            req.name = item.name
            req.action_id = item.action_id
            req.parameters = copy.deepcopy(item.parameters)
            resp = check_precondition(req)
            return resp.success
        except rospy.ServiceException as e:
            print("process_action_effect service call failed: %s" %e)
            return False   
register(
    id='MiniGrid-MultiHiPRLGridShortPose-v0',
    entry_point='gym_minigrid.envs:MultiHiPRLGridShortPoseV0'
)

"""
print(self.observation_space)
        data_path = '/home/morin/catkin_ws/src/hiprl_replicate/pddl/'
        domain_path = data_path + 'hiprl_mini.pddl'
        problem0_path = data_path + 'hiprl_problem0.pddl'
        problem_path = data_path + 'problem.pddl'
        planner_command = '/home/morin/catkin_ws/src/rosplan/rosplan_planning_system/common/bin/popf2 DOMAIN PROBLEM /home/morin/catkin_ws/src/hiprl_replicate/pddl/plan0.pddl'
        
        # objmap_to_image converter launch
        file0_package = 'find_frontier'
        file0_executable = 'objmap_to_image_converter'
        node0 = roslaunch.core.Node(file0_package, file0_executable, name = 'objmap_to_image_converter' + str(process_num), args = str(process_num), output='screen')
        self.launch = roslaunch.scriptapi.ROSLaunch()
        self.launch.start()
        self.process0 = self.launch.launch(node0)
        
        # hiprl_explore launch
        file1_package = 'find_frontier'
        file1_executable = 'find_frontier_node'
        node1 = roslaunch.core.Node(file1_package, file1_executable, name = 'hiprl_explore' + str(process_num), args = str(process_num), output='screen')
        f = open('/home/morin/catkin_ws/src/find_frontier/param/global_costmap' + str(process_num) + '.yaml', 'r')
        yaml_file = load(f)
        f.close()
        upload_params('/hiprl_explore' + str(process_num) + '/', yaml_file)
        self.process1 = self.launch.launch(node1)
        rospy.set_param('use_sim_time', False)
        
        # knowledge manager launch
        file2_package = 'rosplan_knowledge_base'
        file2_executable = 'knowledgeBase'
        node2 = roslaunch.core.Node(file2_package, file2_executable, name = 'rosplan_knowledge_base' + str(process_num), args = str(process_num), output='screen')
        rospy.set_param('rosplan_knowledge_base' + str(process_num) + '/domain_path', domain_path)
        rospy.set_param('rosplan_knowledge_base' + str(process_num) + '/problem_path', problem0_path)
        rospy.set_param('rosplan_knowledge_base' + str(process_num) + '/use_unknowns', False)
        self.process2 = self.launch.launch(node2)
        
        file3_package = 'rosplan_planning_system'
        file3_executable = 'problemInterface'
        node3 = roslaunch.core.Node(file3_package, file3_executable, name = 'rosplan_problem_interface' + str(process_num), args = str(process_num), output='screen')
        rospy.set_param('rosplan_problem_interface' + str(process_num) + '/knowledge_base', 'rosplan_knowledge_base' + str(process_num))
        rospy.set_param('rosplan_problem_interface' + str(process_num) + '/domain_path', domain_path)
        rospy.set_param('rosplan_problem_interface' + str(process_num) + '/problem_path', problem_path)
        rospy.set_param('rosplan_problem_interface' + str(process_num) + '/problem_topic', 'problem_instance')
        self.process3 = self.launch.launch(node3)
        
        file4_package = 'hiprl_replicate'
        file4_executable = 'knowledge_update_node'
        node4 = roslaunch.core.Node(file4_package, file4_executable, name = 'rosplan_knowledge_update_node' + str(process_num), args = str(process_num), output='screen')
        rospy.set_param('rosplan_knowledge_update_node' + str(process_num) + '/knowledge_base', 'rosplan_knowledge_base' + str(process_num))
        self.process4 = self.launch.launch(node4)
        
        # planner launch
        file5_package = 'rosplan_planning_system'
        file5_executable = 'popf_planner_interface'
        node5 = roslaunch.core.Node(file5_package, file5_executable, name = 'rosplan_planner_interface' + str(process_num), args = str(process_num), output='screen')
        rospy.set_param('rosplan_planner_interface' + str(process_num) + '/use_problem_topic', True)
        rospy.set_param('rosplan_planner_interface' + str(process_num) + '/problem_topic', 'rosplan_problem_interface' + str(process_num) + '/problem_instance')
        rospy.set_param('rosplan_planner_interface' + str(process_num) + '/planner_topic', 'planner_output')
        rospy.set_param('rosplan_planner_interface' + str(process_num) + '/domain_path', domain_path)
        rospy.set_param('rosplan_planner_interface' + str(process_num) + '/problem_path', problem_path)
        rospy.set_param('rosplan_planner_interface' + str(process_num) + '/data_path', data_path)
        rospy.set_param('rosplan_planner_interface' + str(process_num) + '/planner_interface', 'popf_planner_interface')
        rospy.set_param('rosplan_planner_interface' + str(process_num) + '/planner_command', planner_command)
        self.process5 = self.launch.launch(node5)
                
        file6_package = 'rosplan_planning_system'
        file6_executable = 'pddl_simple_plan_parser'
        node6 = roslaunch.core.Node(file6_package, file6_executable, name = 'rosplan_parsing_interface' + str(process_num), args = str(process_num), output='screen')
        rospy.set_param('rosplan_parsing_interface' + str(process_num) + '/knowledge_base', 'rosplan_knowledge_base' + str(process_num))
        rospy.set_param('rosplan_parsing_interface' + str(process_num) + '/planner_topic', 'rosplan_planner_interface' + str(process_num) + '/planner_output')
        rospy.set_param('rosplan_parsing_interface' + str(process_num) + '/plan_topic', 'complete_plan')
        self.process6 = self.launch.launch(node6)
"""

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
        
"""
    def dispatch_plan(self, actions, render = False):
        print("dispatch plan called")
        reward_sum = 0
        obs, reward, done, info = None, None, None, {'fwd_cell': None}
        if self.dispatch_plan_action_id - self.prev_dispatch_plan_action_id > 0 and len(actions) > 0:
            precondition_check = self.check_precondition(actions[self.dispatch_plan_action_id][0])
            if precondition_check == False:
                print("dispatch_plan: {} precondition not achieved".format(actions[self.dispatch_plan_action_id][0].name))
                self.dispatch_plan_action_id = 0
                self.prev_dispatch_plan_action_id = -1
                return None, None, None, None
        # dispatch plan until potential goal update
        # 'actions[self.dispatch_plan_action_id][2] == False' means it is not potential goal update related semantic action
        print("actions length: {}, dispatch_plan_id: {}".format(len(actions), self.dispatch_plan_action_id))
        print("actions self.dispatch_plan_action_id length: {}".format(len(actions[self.dispatch_plan_action_id])))

        while actions[self.dispatch_plan_action_id][2] == False:
            print("actions length: {}, dispatch_plan_id: {}".format(len(actions), self.dispatch_plan_action_id))
            print("actions self.dispatch_plan_action_id length: {}".format(len(actions[self.dispatch_plan_action_id])))

            while len(actions[self.dispatch_plan_action_id][1]) > 0:    
                action = actions[self.dispatch_plan_action_id][1].pop(0)
                obs, reward, done, info = self.step(action)
                if render:
                    self.render()
                reward_sum += reward
            self.prev_dispatch_plan_action_id = self.dispatch_plan_action_id        
    
            # if actions for semantic action is done (= actions[dispatch_plan_action_id][1] is empty)
            if not actions[self.dispatch_plan_action_id][1]:
                self.process_action_effect(actions[self.dispatch_plan_action_id][0])
                if len(actions)-1 > self.dispatch_plan_action_id:
                    self.dispatch_plan_action_id += 1
                else:
                    break
        
        if self.dispatch_plan_action_id + 1 <= len(actions):
            while len(actions[self.dispatch_plan_action_id][1]) > 0:    
                action = actions[self.dispatch_plan_action_id][1].pop(0)
                obs, reward, done, info = self.step(action)
                if render:
                    self.render()
                reward_sum += reward
            self.prev_dispatch_plan_action_id = self.dispatch_plan_action_id        
        
            # if actions for semantic action is done (= actions[dispatch_plan_action_id][1] is empty)
            if not actions[self.dispatch_plan_action_id][1]:
                self.process_action_effect(actions[self.dispatch_plan_action_id][0])
                if len(actions)-1 > self.dispatch_plan_action_id:
                    self.dispatch_plan_action_id += 1
        
        return obs, reward_sum, done, info
"""
"""
    def dispatch_plan(self, actions, render = False):
        print("dispatch plan called")
        reward_sum = 0
        obs, reward, done, info = None, None, None, {'fwd_cell': None}
        if self.dispatch_plan_action_id - self.prev_dispatch_plan_action_id > 0 and len(actions) > 0:
            precondition_check = self.check_precondition(actions[self.dispatch_plan_action_id][0])
            if precondition_check == False:
                print("dispatch_plan: {} precondition not achieved".format(actions[self.dispatch_plan_action_id][0].name))
                self.dispatch_plan_action_id = 0
                self.prev_dispatch_plan_action_id = -1
                return None, None, None, None
        # dispatch plan until potential goal update
        # 'actions[self.dispatch_plan_action_id][2] == False' means it is not potential goal update related semantic action
        #print("actions length: {}, dispatch_plan_id: {}".format(len(actions), self.dispatch_plan_action_id))
        #print("actions self.dispatch_plan_action_id length: {}".format(len(actions[self.dispatch_plan_action_id])))

        if len(actions[self.dispatch_plan_action_id][1]) > 0:    
            action = actions[self.dispatch_plan_action_id][1].pop(0)
            obs, reward, done, info = self.step(action)
            if render:
                self.render()
            reward_sum += reward
            self.prev_dispatch_plan_action_id = self.dispatch_plan_action_id        

        # if actions for semantic action is done (= actions[dispatch_plan_action_id][1] is empty)
        if not actions[self.dispatch_plan_action_id][1]:
            self.process_action_effect(actions[self.dispatch_plan_action_id][0])
            if len(actions)-1 > self.dispatch_plan_action_id:
                self.dispatch_plan_action_id += 1
            
        
        return obs, reward_sum, done, info
        
        
    # plan 2021 12 08    
    def plan(self, id_list):
        print("Updating knowledge for planning-only agents")
        rospy.wait_for_service('/planning_agent_update'+ str(self.process_num))
        try:
            planning_agent_update = rospy.ServiceProxy('/planning_agent_update' + str(self.process_num), PlanningID)            
            resp = planning_agent_update(id_list)
        except rospy.ServiceException as e:
            print("Planning Agent Update Service call failed: %s" %e)
            return None
        
        print("Allocating tasks for planning-only agents")
        rospy.wait_for_service('/task_allocate'+ str(self.process_num))
        try:
            task_allocate = rospy.ServiceProxy('/task_allocate' + str(self.process_num), PlanningID)
            resp = task_allocate(id_list)
        except rospy.ServiceException as e:
            print("Planning Agent Update Service call failed: %s" %e)
            return None
        
        self.complete_plan_id_list = set([])
        for i in id_list:
            print("Generating a Problem for each agent")
            rospy.wait_for_service('/rosplan_problem_interface'+ str(self.process_num) + '_' + str(i) + '/problem_generation_server')
            try:
                problem_generation = rospy.ServiceProxy('/rosplan_problem_interface' + str(self.process_num) + '_' + str(i) + '/problem_generation_server', Empty)
                resp = problem_generation()
            except rospy.ServiceException as e:
                print("Problem Generation Service call failed: %s" %e)
                return None

            print("Planning for each agent")            
            rospy.wait_for_service('/rosplan_planner_interface' + str(self.process_num) + '_' + str(i) +'/planning_server')
            try:
                run_planner = rospy.ServiceProxy('/rosplan_planner_interface' + str(self.process_num) + '_' + str(i) + '/planning_server', Empty)
                resp = run_planner()

            except rospy.ServiceException as e:
                print("Planning Service call failed: %s" %e)
                return None

            print("Executing the Plan")                        
            rospy.wait_for_service('/rosplan_parsing_interface' + str(self.process_num) + '_' + str(i) + '/parse_plan')
            try:
                parse_plan = rospy.ServiceProxy('/rosplan_parsing_interface' + str(self.process_num) + '_' + str(i) + '/parse_plan', Empty)
                resp = parse_plan()
            except rospy.ServiceException as e:
                print("Plan Parsing Service call failed: %s" %e)
                return None
        #rospy.wait_for_service('/rosplan_parsing_interface/alert_plan_action_num')
        #try:
        #    alert_plan_action_num = rospy.ServiceProxy('/rosplan_parsing_interface/alert_plan_action_num', AlertPlanActionNum)
        #    resp = alert_plan_action_num()
        #    self.planner_action_num = resp.num_actions
        #except rospy.ServiceException as e:
        #    print("Plan Parsing Service call failed: %s" %e)
        counter = 0
        while(len(self.complete_plan_id_list & set(id_list)) != len(id_list)):
            time.sleep(0.1)
            counter = counter + 1
            print("complete_plan_flag: ", self.action_execution_flag)
            if counter > 10:
                self.dispatch_plan_action_id = 0
                self.prev_dispatch_plan_action_id = -1
                return None
        print("complete_plan_flag is set to True")
        
        self.generate_actions_from_complete_plan(id_list)
        self.dispatch_plan_action_id = 0
        self.prev_dispatch_plan_action_id = -1
        self.complete_plan_flag = False
        return     
        """