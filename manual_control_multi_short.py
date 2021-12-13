#!/usr/bin/env python3
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

key_array = []
opt_array = []
def redraw(img):
    if not args.agent_view:
        img = env.render('rgb_array', tile_size=args.tile_size)

    window.show_img(img)

def reset():
    global key_array, opt_array

    if args.seed != -1:
        env.seed(args.seed)

    obs = env.reset()
    key_array = []
    opt_array = []
    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    redraw(obs)

def step(action):
    obs, reward, done, info = env.step(action)
    print('step={}, reward={}, action={}'.format(env.step_count, reward, action))

    if done:
        print('done!')
        reset()
    else:
        redraw(obs)

def invoke(meta_action):
    reward_sum = 0
    done = False
    info = {'fwd_cell': None}
    explore_agent_id_list = []
    plan_agent_id_list = []
    scan_agent_id_list = []
    for i in range(len(meta_action)):
        if meta_action[i] == env.meta_actions.explore:
            explore_agent_id_list.append(i)
            env.agents[i].mode = env.Option_mode.explore
            
        elif meta_action[i] == env.meta_actions.plan:
            plan_agent_id_list.append(i)
            env.agents[i].mode = env.Option_mode.plan
            
        elif meta_action[i] == env.meta_actions.scan:
            scan_agent_id_list.append(i)
            env.agents[i].mode = env.Option_mode.scan
      
    if len(explore_agent_id_list) > 0:
        env.explore(explore_agent_id_list)
    
    if len(plan_agent_id_list) > 0:
        env.plan(plan_agent_id_list)
    
    if len(scan_agent_id_list) > 0:
        env.scan(scan_agent_id_list)
    
    min_option_len = 1000  # 1000 is nothing special, just for simple infinity value
    for i in range(env.num_agents):
        if len(env.agents[i].action_list) < min_option_len:
            min_option_len = len(env.agents[i].action_list)
    
    # Execute generated action sequences         
    for i in range(min_option_len):
        action = []
        for j in range(env.num_agents):
            action.append(env.agents[j].action_list.pop(0))
        obs, reward, done, info = env.step(action)
        reward_sum += reward
        print('step={}, reward={}, action={}'.format(env.step_count, reward, action))
        if done:
            print('done!')
            reset()
            break
        else:
            redraw(obs)
            time.sleep(0.5)
 
    print('%s, Overall reward=%.2f' % (meta_action, reward_sum))

def key_handler(event):
    global key_array, opt_array

    print('pressed', event.key)

    print('current key_array: {}'.format(key_array))

    print('current opt_array: {}'.format(opt_array))

    if event.key == 'escape':
        window.close()
        return

    elif event.key == 'backspace':
        reset()
        return

    elif event.key == 'left':
        key_array.append(env.actions.left)

    elif event.key == 'right':
        key_array.append(env.actions.right)

    elif event.key == 'up':
        key_array.append(env.actions.forward)

    # Spacebar
    elif event.key == ' ':
        key_array.append(env.actions.open)
    elif event.key == 'c':
        key_array.append(env.actions.close)
    elif event.key == 'r':
        opt_array.append(env.meta_actions.scan)
    elif event.key == 'e':
        opt_array.append(env.meta_actions.explore)
        
    if event.key == 'p':
        opt_array.append(env.meta_actions.plan)
    #    invoke(env.meta_actions.plan)
    #    return
    #if event.key == 'k':
    #    invoke(env.meta_actions.keep_previous)
    #    return
    
    
    elif event.key == 'pageup':
        key_array.append(env.actions.pickup)

    elif event.key == 'pagedown':
        key_array.append(env.actions.drop)

    elif event.key == 'enter':
        key_array.append(env.actions.done)
    
    print('pressed', event.key)

    print('current key_array: {}'.format(key_array))

    print('current opt_array: {}'.format(opt_array))

    
    if len(key_array) == num_agents:
        step(key_array)
        key_array = []
        print('Reset key_array: {}'.format(key_array))


    if len(opt_array) == num_agents:
        invoke(opt_array)
        opt_array = []
        print('Reset opt_array: {}'.format(opt_array))


    


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

parser.add_argument(
    '--num_envs',
    type = int,
    default = 0,
    help="number of environments to run in parallel - 1")

parser.add_argument(
    '--num_agents',
    type = int,
    default = 1,
    help="number of agents to run")


args = parser.parse_args()
env = gym.make(args.env, process_num = args.num_envs, num_agents = args.num_agents)
num_agents = args.num_agents
if args.agent_view:
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

window = Window('gym_minigrid - ' + args.env)
window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window.show(block=True)