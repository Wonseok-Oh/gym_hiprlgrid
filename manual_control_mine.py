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

def redraw(img):
    if not args.agent_view:
        img = env.render('rgb_array', tile_size=args.tile_size)
    print(img.shape)
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
    print("current_mode: {}".format(env.mode))
    if meta_action == env.meta_actions.keep_previous:
        if env.mode == env.Option_mode.init:
            reward_sum = -0.5
            done = True
        
        elif env.mode == env.Option_mode.explore:
            if len(env.explore_action_list) <= 0:
                env.explore_action_list = env.explore()
            if len(env.explore_action_list) > 0:
                action = env.explore_action_list.pop(0)
                obs, reward, done, info = env.step(action)
                print('step=%s, reward=%.2f' % (env.step_count, reward))
                reward_sum += reward

                if done:
                    print('done!')
                    reset()
                else:
                    redraw(obs)

            
        elif env.mode == env.Option_mode.scan:
            obs, reward, done, info = env.step(env.Actions.left)
            print('step=%s, reward=%.2f' % (env.step_count, reward))
            reward_sum += reward
            if done:
                print('done!')
                reset()
            else:
                redraw(obs)
            
        elif env.mode == env.Option_mode.plan:
            if env.plan_action_list == None or len(env.plan_action_list) == 0:
                return invoke(env.meta_actions.plan)
            else:
                len_actions = 0
                for i in range(len(env.plan_action_list)):
                    len_actions = len_actions + len(env.plan_action_list[i][1])
                if len_actions > 0:
                    obs, reward, done, info = env.dispatch_plan(env.plan_action_list)
                    if reward == None: # invalid plan
                        obs = env.gen_obs()
                        reward= 0
                        done = False
                        info = {'fwd_cell': None}
                            
                    print('step=%s, reward=%.2f' % (env.step_count, reward))
                    reward_sum += reward
                    if done:
                        print('done!')
                        reset()
                    else:
                        redraw(obs)
                        
    elif meta_action == env.meta_actions.scan:
        env.mode= env.Option_mode.scan
        obs, reward, done, info = env.step(env.Actions.left)
        print('step=%s, reward=%.2f' % (env.step_count, reward))
        reward_sum += reward
        if done:
            print('done!')
            reset()
        else:
            redraw(obs)
                
    elif meta_action == env.meta_actions.explore:
        env.mode = env.Option_mode.explore
        env.explore_action_list = env.explore()
        if len(env.explore_action_list) > 1:
            action = env.explore_action_list.pop(0)
            obs, reward, done, info = env.step(action)
            print('step=%s, reward=%.2f' % (env.step_count, reward))
            reward_sum += reward

            if done:
                print('done!')
                reset()
            else:
                redraw(obs)
    
    elif meta_action == env.meta_actions.plan:
        env.mode = env.Option_mode.plan
        env.plan_action_list = env.plan()
    
        if env.plan_action_list == None or len(env.plan_action_list) == 0:
            obs, reward, done, info = env.step(env.Actions.pickup)

            done = True
            print('step=%s, reward=%.2f' % (env.step_count, reward))
            reward_sum += reward
            if done:
                print('done!')
                reset()
            else:
                redraw(obs)
        
        else:
            len_actions = 0
            for i in range(len(env.plan_action_list)):
                len_actions = len_actions + len(env.plan_action_list[i][1])
            if len_actions > 0:
                obs, reward, done, info = env.dispatch_plan(env.plan_action_list)

                if reward == None: # invalid plan
                    obs = env.gen_obs()
                    reward= 0
                    done = False
                    info = {'fwd_cell': None}
                   
                print('step=%s, reward=%.2f' % (env.step_count, reward))
                reward_sum += reward
            if done:
                print('done!')
                reset()
            else:
                redraw(obs)
            
        #process.stop()
    #print('%s, Overall reward=%.2f' % (meta_action, reward_sum))

def key_handler(event):
    print('pressed', event.key)

    if event.key == 'escape':
        window.close()
        return

    if event.key == 'backspace':
        reset()
        return

    if event.key == 'left':
        step(env.actions.left)
        return
    if event.key == 'right':
        step(env.actions.right)
        return
    if event.key == 'up':
        step(env.actions.forward)
        return

    # Spacebar
    if event.key == ' ':
        step(env.actions.open)
        return
    if event.key == 'c':
        step(env.actions.close)
        return
    if event.key == 'r':
        invoke(env.meta_actions.scan)
        return
    if event.key == 'e':
        invoke(env.meta_actions.explore)
        return
    if event.key == 'p':
        invoke(env.meta_actions.plan)
        return
    if event.key == 'k':
        invoke(env.meta_actions.keep_previous)
        return
    
    
    if event.key == 'pageup':
        step(env.actions.pickup)
        return
    if event.key == 'pagedown':
        step(env.actions.drop)
        return

    if event.key == 'enter':
        step(env.actions.done)
        return

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
    help="number of environments to run in parallel")

args = parser.parse_args()
env = gym.make(args.env, process_num = args.num_envs)
if args.agent_view:
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

window = Window('gym_minigrid - ' + args.env)
window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window.show(block=True)