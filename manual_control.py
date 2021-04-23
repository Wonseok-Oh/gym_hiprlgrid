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
        actions = env.explore()
        for i in range(len(actions)):
            obs, reward, done, info = env.step(actions[i])
            print('step=%s, reward=%.2f' % (env.step_count, reward))
            reward_sum += reward
            if done:
                print('done!')
                reset()
            else:
                redraw(obs)
                time.sleep(1)
        
        #process.stop()
    print('%s, Overall reward=%.2f' % (meta_action, reward_sum))

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

args = parser.parse_args()
env = gym.make(args.env)

if args.agent_view:
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

window = Window('gym_minigrid - ' + args.env)
window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window.show(block=True)