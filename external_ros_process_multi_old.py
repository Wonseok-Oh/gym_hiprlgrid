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
   
    # hiprl_explore launch
    file1_package = 'find_frontier'
    file1_executable = 'find_frontier_multi_node'
    node1 = roslaunch.core.Node(file1_package, file1_executable, name = 'hiprl_explore' + str(process_num), args = str(process_num), output='screen')
    f = open('/home/morin/catkin_ws/src/find_frontier/param/multi/global_costmap' + str(process_num) + '.yaml', 'r')
    yaml_file = load(f)
    f.close()
    upload_params('/hiprl_explore' + str(process_num) + '/', yaml_file)
    process1 = launch.launch(node1)
    rospy.set_param('use_sim_time', False)
    
    # Central knowledge base launch
    file2_package = 'rosplan_knowledge_base'
    file2_executable = 'knowledgeBase'
    node2 = roslaunch.core.Node(file2_package, file2_executable, name = 'rosplan_knowledge_base' + str(process_num), args = str(process_num), output='screen')
    rospy.set_param('rosplan_knowledge_base' + str(process_num) + '/domain_path', domain_path)
    rospy.set_param('rosplan_knowledge_base' + str(process_num) + '/problem_path', problem0_path)
    rospy.set_param('rosplan_knowledge_base' + str(process_num) + '/use_unknowns', False)
    process2 = launch.launch(node2)

    # Central problem generator launch --> no need    
    file3_package = 'rosplan_planning_system'
    file3_executable = 'problemInterface'
    node3 = roslaunch.core.Node(file3_package, file3_executable, name = 'rosplan_problem_interface' + str(process_num), args = str(process_num), output='screen')
    rospy.set_param('rosplan_problem_interface' + str(process_num) + '/knowledge_base', 'rosplan_knowledge_base' + str(process_num))
    rospy.set_param('rosplan_problem_interface' + str(process_num) + '/domain_path', domain_path)
    rospy.set_param('rosplan_problem_interface' + str(process_num) + '/problem_path', problem_path)
    rospy.set_param('rosplan_problem_interface' + str(process_num) + '/problem_topic', 'problem_instance')
    process3 = launch.launch(node3)

    # Central knowledge manager launch --> only for recording box(building) locations
    file4_package = 'hiprl_replicate'
    file4_executable = 'knowledge_update_multi_node'
    node4 = roslaunch.core.Node(file4_package, file4_executable, name = 'rosplan_knowledge_update_multi_node' + str(process_num), args = str(process_num), output='screen')
    rospy.set_param('rosplan_knowledge_update_multi_node' + str(process_num) + '/knowledge_base', 'rosplan_knowledge_base' + str(process_num))
    process4 = launch.launch(node4)

    # agents' individual knowledge managers launch
    file2_, file3_, file4_, file5_, file6_ = ([None] * num_agents,) * 5
    node2_, node3_, node4_, node5_, node6_ = ([None] * num_agents,) * 5
    process2_, process3_, process4_, process5_, process6_ = ([None] * num_agents,) * 5
    file2_package_, file3_package_, file4_package_, file5_package_, file6_package_ = ([None] * num_agents,) * 5
    file2_executable_, file3_executable_, file4_executable_, file5_executable_, file6_executable_  = ([None] * num_agents,) * 5
    
    for i in range(int(num_agents)):
        print("agent {}'s plan/knowledge modules init".format(i))
        # Knolwedgebase for agent i
        file2_package_[i] = 'rosplan_knowledge_base'
        file2_executable_[i] = 'knowledgeBase'
        node2_[i] = roslaunch.core.Node(file2_package_[i], file2_executable_[i], name = 'rosplan_knowledge_base' + str(process_num) + '_' + str(i), args = str(process_num) + '_' + str(i), output='screen')
        rospy.set_param('rosplan_knowledge_base' + str(process_num) + '_' + str(i) + '/domain_path', domain_path)
        rospy.set_param('rosplan_knowledge_base' + str(process_num) + '_' + str(i) + '/problem_path', problem0_path)
        rospy.set_param('rosplan_knowledge_base' + str(process_num) + '_' + str(i) + '/use_unknowns', False)
        process2_[i] = launch.launch(node2_[i])
        
        # Problem generator for agent i
        file3_package_[i] = 'rosplan_planning_system'
        file3_executable_[i] = 'problemInterface'
        node3_[i] = roslaunch.core.Node(file3_package_[i], file3_executable_[i], name = 'rosplan_problem_interface' + str(process_num) + '_' + str(i), args = str(process_num) + '_' + str(i), output='screen')
        rospy.set_param('rosplan_problem_interface' + str(process_num) + '_' + str(i) + '/knowledge_base', 'rosplan_knowledge_base' + str(process_num))
        rospy.set_param('rosplan_problem_interface' + str(process_num) + '_' + str(i) + '/domain_path', domain_path)
        rospy.set_param('rosplan_problem_interface' + str(process_num) + '_' + str(i) + '/problem_path', problem_path)
        rospy.set_param('rosplan_problem_interface' + str(process_num) + '_' + str(i) + '/problem_topic', 'problem_instance')
        process3_[i] = launch.launch(node3_[i])
        
        # Knowledge manager for agent i
        file4_package_[i] = 'hiprl_replicate'
        file4_executable_[i] = 'knowledge_update_multi_individual_node'
        node4_[i] = roslaunch.core.Node(file4_package_[i], file4_executable_[i], name = 'rosplan_knowledge_update_multi_individual_node' + str(process_num) + '_' + str(i), args = str(process_num) + '_' + str(i), output='screen')
        rospy.set_param('rosplan_knowledge_update_multi_inidividual_node' + str(process_num) + '_' + str(i) + '/knowledge_base', 'rosplan_knowledge_base' + str(process_num) + '_' + str(i))
        rospy.set_param('rosplan_knowledge_update_multi_inidividual_node' + str(process_num) + '_' + str(i) + '/total_agent_num', num_agents)
        process4_[i] = launch.launch(node4_[i])

        # Planner for agent i
        file5_package_[i] = 'rosplan_planning_system'
        file5_executable_[i] = 'popf_planner_interface'
        node5_[i] = roslaunch.core.Node(file5_package_[i], file5_executable_[i], name = 'rosplan_planner_interface' + str(process_num) + '_' + str(i), args = str(process_num) + '_' + str(i), output='screen')
        rospy.set_param('rosplan_planner_interface' + str(process_num) + '_' + str(i) + '/use_problem_topic', True)
        rospy.set_param('rosplan_planner_interface' + str(process_num) + '_' + str(i) + '/problem_topic', 'rosplan_problem_interface' + str(process_num) + '/problem_instance')
        rospy.set_param('rosplan_planner_interface' + str(process_num) + '_' + str(i) + '/planner_topic', 'planner_output')
        rospy.set_param('rosplan_planner_interface' + str(process_num) + '_' + str(i) + '/domain_path', domain_path)
        rospy.set_param('rosplan_planner_interface' + str(process_num) + '_' + str(i) + '/problem_path', problem_path)
        rospy.set_param('rosplan_planner_interface' + str(process_num) + '_' + str(i) + '/data_path', data_path)
        rospy.set_param('rosplan_planner_interface' + str(process_num) + '_' + str(i) + '/planner_interface', 'popf_planner_interface')
        rospy.set_param('rosplan_planner_interface' + str(process_num) + '_' + str(i) + '/planner_command', planner_command)
        process5_[i] = launch.launch(node5_[i])
                
        # Plan parser for agent i
        file6_package_[i] = 'rosplan_planning_system'
        file6_executable_[i] = 'pddl_simple_plan_parser'
        node6_[i] = roslaunch.core.Node(file6_package_[i], file6_executable_[i], name = 'rosplan_parsing_interface' + str(process_num) + '_' + str(i), args = str(process_num) + '_' + str(i), output='screen')
        rospy.set_param('rosplan_parsing_interface' + str(process_num) + '_' + str(i) + '/knowledge_base', 'rosplan_knowledge_base' + str(process_num))
        rospy.set_param('rosplan_parsing_interface' + str(process_num) + '_' + str(i) + '/planner_topic', 'rosplan_planner_interface' + str(process_num) + '/planner_output')
        rospy.set_param('rosplan_parsing_interface' + str(process_num) + '_' + str(i) + '/plan_topic', 'complete_plan')
        process6_[i] = launch.launch(node6_[i])


    
    # Central planner launch --> no need
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
    
    # Central plan parser launch --> no need
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