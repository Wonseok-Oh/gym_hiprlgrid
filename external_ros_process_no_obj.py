import sys
import rospy
import roslaunch
from rosparam import upload_params
from yaml import load

if __name__ == '__main__':
    process_num = sys.argv[1]
    
    if len(sys.argv) < 2:
        print("Insufficient arguments")
        sys.exit()
        
    print(process_num)
    data_path = '/home/morin/catkin_ws/src/hiprl_replicate/pddl/'
    domain_path = data_path + 'hiprl_mini.pddl'
    problem0_path = data_path + 'hiprl_problem0.pddl'
    problem_path = data_path + 'problem' + str(process_num) + '.pddl'

    planner_command = '/home/morin/catkin_ws/src/rosplan/rosplan_planning_system/common/bin/popf2 DOMAIN PROBLEM /home/morin/catkin_ws/src/hiprl_replicate/pddl/plan0.pddl'
    
    # objmap_to_image converter launch
#    file0_package = 'find_frontier'
#    file0_executable = 'objmap_to_image_converter'
#    node0 = roslaunch.core.Node(file0_package, file0_executable, name = 'objmap_to_image_converter' + str(process_num), args = str(process_num), output='screen')

#    process0 = launch.launch(node0)


    launch = roslaunch.scriptapi.ROSLaunch()
    launch.start()    
    # hiprl_explore launch
    file1_package = 'find_frontier'
    file1_executable = 'find_frontier_node'
    node1 = roslaunch.core.Node(file1_package, file1_executable, name = 'hiprl_explore' + str(process_num), args = str(process_num), output='screen')
    f = open('/home/morin/catkin_ws/src/find_frontier/param/global_costmap' + str(process_num) + '.yaml', 'r')
    yaml_file = load(f)
    f.close()
    upload_params('/hiprl_explore' + str(process_num) + '/', yaml_file)
    process1 = launch.launch(node1)
    rospy.set_param('use_sim_time', False)
    
    # knowledge manager launch
    file2_package = 'rosplan_knowledge_base'
    file2_executable = 'knowledgeBase'
    node2 = roslaunch.core.Node(file2_package, file2_executable, name = 'rosplan_knowledge_base' + str(process_num), args = str(process_num), output='screen')
    rospy.set_param('rosplan_knowledge_base' + str(process_num) + '/domain_path', domain_path)
    rospy.set_param('rosplan_knowledge_base' + str(process_num) + '/problem_path', problem0_path)
    rospy.set_param('rosplan_knowledge_base' + str(process_num) + '/use_unknowns', False)
    process2 = launch.launch(node2)
    
    file3_package = 'rosplan_planning_system'
    file3_executable = 'problemInterface'
    node3 = roslaunch.core.Node(file3_package, file3_executable, name = 'rosplan_problem_interface' + str(process_num), args = str(process_num), output='screen')
    rospy.set_param('rosplan_problem_interface' + str(process_num) + '/knowledge_base', 'rosplan_knowledge_base' + str(process_num))
    rospy.set_param('rosplan_problem_interface' + str(process_num) + '/domain_path', domain_path)
    rospy.set_param('rosplan_problem_interface' + str(process_num) + '/problem_path', problem_path)
    rospy.set_param('rosplan_problem_interface' + str(process_num) + '/problem_topic', 'problem_instance')
    process3 = launch.launch(node3)
    
    file4_package = 'hiprl_replicate'
    file4_executable = 'knowledge_update_node'
    node4 = roslaunch.core.Node(file4_package, file4_executable, name = 'rosplan_knowledge_update_node' + str(process_num), args = str(process_num), output='screen')
    rospy.set_param('rosplan_knowledge_update_node' + str(process_num) + '/knowledge_base', 'rosplan_knowledge_base' + str(process_num))
    process4 = launch.launch(node4)
    
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
    process5 = launch.launch(node5)
            
    file6_package = 'rosplan_planning_system'
    file6_executable = 'pddl_simple_plan_parser'
    node6 = roslaunch.core.Node(file6_package, file6_executable, name = 'rosplan_parsing_interface' + str(process_num), args = str(process_num), output='screen')
    rospy.set_param('rosplan_parsing_interface' + str(process_num) + '/knowledge_base', 'rosplan_knowledge_base' + str(process_num))
    rospy.set_param('rosplan_parsing_interface' + str(process_num) + '/planner_topic', 'rosplan_planner_interface' + str(process_num) + '/planner_output')
    rospy.set_param('rosplan_parsing_interface' + str(process_num) + '/plan_topic', 'complete_plan')
    process6 = launch.launch(node6)
    try: 
        launch.spin()
    finally:
        # After Ctrl+C, stop all nodes from running
        launch.shutdown()