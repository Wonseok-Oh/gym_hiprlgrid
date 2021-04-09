import rospy
from geometry_msgs.msg import PoseStamped
import sys, select, termios, tty
import time
import argparse
import numpy as np
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window
import roslaunch
import rospy
from geometry_msgs.msg import PoseStamped

def redraw(img):
    if not args.agent_view:
        img = env.render('rgb_array', tile_size=args.tile_size)

    window.show_img(img)

def reset():
    if args.seed != -1:
        env.seed(args.seed)

    obs = env.reset()

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    redraw(obs)

def step(action):
    obs, reward, done, info = env.step(action)
    print('step=%s, reward=%.2f' % (env.step_count, reward))

    if done:
        print('done!')
        reset()
    else:
        redraw(obs)

def invoke(meta_action):
    reward_sum = 0
    if meta_action == env.meta_actions.scan:
        for i in range(4):
            obs, reward, done, info = env.step(env.actions.left)
            print('step=%s, reward=%.2f' % (env.step_count, reward))
            reward_sum += reward
            if done:
                print('done!')
                reset()
            else:
                redraw(obs)
                time.sleep(1)
                
    if meta_action == env.meta_actions.explore:
        package = 'find_frontier'
        executable = 'find_frontier_node'
        node = roslaunch.core.Node(package, executable)
        
        launch = roslaunch.scriptapi.ROSLaunch()
        launch.start()
        
        process = launch.launch(node)
        print(process.is_alive())
        time.sleep(10)
        process.stop()
    print('%s, Overall reward=%.2f' % (meta_action, reward_sum))

def getKey():
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
        
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        help="gym environment to load",
        default='MiniGrid-MultiRoom-N6-v0'
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=-1
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        help="size at which to render tiles",
        default=32
    )
    parser.add_argument(
        '--agent_view',
        default=False,
        help="draw the agent sees (partially observable view)",
        action='store_true'
    )

    args = parser.parse_args()
    env = gym.make(args.env)
    if args.agent_view:
        env = RGBImgPartialObsWrapper(env)
        env = ImgObsWrapper(env)

    window = Window('gym_minigrid - ' + args.env)
    reset()
    
    settings = termios.tcgetattr(sys.stdin)
    pub = rospy.Publisher('initial_pose', PoseStamped, queue_size = 1)
    try:
        while(1):
            key = getKey()
            if key == 'w':
                step(env.actions.forward)
            
            elif key == 'a':
                step(env.actions.left)
                
            elif key == 'd':
                step(env.actions.right)
            
            elif key == ' ':
                step(env.actions.open)
            
            elif key == 'c':
                step(env.actions.close)
                
            elif key == 'r':
                invoke(env.meta_actions.scan)
                
            elif key == 'e':
                invoke(env.meta_actions.explore)
                
            elif key == 'p':
                invoke(env.meta_actions.plan)
                
            elif key == ',':
                step(env.actions.pickup)
                
            elif key == '.':
                step(env.actions.drop)
            
            elif key == '\x03':
                break
    except rospy.ROSInterruptException:
        pass
    
    finally:
        pub.publish(env.agent_init_pos)