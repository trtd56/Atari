#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym, argparse, sys
import numpy as np

from agent import Agent

def main(env_name, monitor=True, load=False, seed=0, gpu=-1):

    env = gym.make(env_name)
    view_path = "./video/" + env_name
    model_path = "./model/" + env_name + "_"

    n_st = env.observation_space.shape[0]
    n_act = env.action_space.n

    agent = Agent(n_act, seed, gpu)
    if load:
        agent.load_model(model_path)

    if monitor:
        env.monitor.start(view_path, video_callable=None, force=True, seed=seed)
    for i_episode in xrange(10000):
        observation = env.reset()
        agent.reset_state(observation)
        ep_end = False
        q_list = []
        r_list = []
        while not ep_end:
            env.render()
            action = agent.act()
            observation, reward, ep_end, _ = env.step(action)
            agent.update_experience(observation, action, reward, ep_end)
            agent.train()
            q_list.append(agent.Q)
            r_list.append(reward)
            #print('%i\t%i\t%f\t%i\t%i\t%f' % (i_episode, agent.step, agent.eps, action, reward, agent.Q))
            if ep_end:
                agent.save_model(model_path)
                break
        print('%i\t%i\t%f\t%i\t%f' % (i_episode, agent.step, agent.eps, sum(r_list), sum(q_list)/float(len(q_list))))
    if monitor:
        env.monitor.close()

if __name__=="__main__":
    #env_name = sys.argv[1]
    #main(env_name)
    parser = argparse.ArgumentParser(description='solve Atari problem.')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()
    env_name = "Breakout-v0"
    main(env_name, monitor=True, load=False, seed=0, gpu=args.gpu)
