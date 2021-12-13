import sys
import rospy
import roslaunch
from rosparam import upload_params
from yaml import load

if __name__ == '__main__':
    process_num = sys.argv[1]
    num_agents = sys.argv[2]
    num_agents = int(num_agents)
    if len(sys.argv) < 3:
        print("Insufficient arguments: process_num num_agents should be given")
        sys.exit()
        
    print("process_num: {}".format(process_num))
    print("num_agents: {}".format(num_agents))
    launch = roslaunch.scriptapi.ROSLaunch()
    launch.start()

    data_path = '/home/morin/catkin_ws/src/hiprl_replicate/pddl/'
    domain_path = data_path + 'hiprl_mini_multi.pddl'
    problem0_path = data_path + 'hiprl_problem0_multi.pddl'
    problem_path = data_path + 'problem' + str(process_num) + '.pddl'
    
    planner_command = '/home/morin/catkin_ws/src/rosplan/rosplan_planning_system/common/bin/popf2 DOMAIN PROBLEM /home/morin/catkin_ws/src/hiprl_replicate/pddl/plan0.pddl'
    
    # objmap_to_image converter launch
    file0_package = 'find_frontier'
    file0_executable = 'objmap_to_image_converter'
    node0 = roslaunch.core.Node(file0_package, file0_executable, name = 'objmap_to_image_converter' + str(process_num), args = str(process_num), output='screen')
    
    process0 = launch.launch(node0)
   
    # explorer / task allocator launch
    file1_package = 'find_frontier'
    file1_executable = 'find_frontier_multi_node'
    node1 = roslaunch.core.Node(file1_package, file1_executable, name = 'hiprl_explore' + str(process_num), args = str(process_num), output='screen')
    f = open('/home/morin/catkin_ws/src/find_frontier/param/multi/global_costmap' + str(process_num) + '.yaml', 'r')
    yaml_file = load(f)
    f.close()
    upload_params('/hiprl_explore' + str(process_num) + '/', yaml_file)
    process1 = launch.launch(node1)
    rospy.set_param('use_sim_time', False)
    
    
    try: 
        launch.spin()
    finally:
        # After Ctrl+C, stop all nodes from running
        launch.shutdown()