#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy, sys
import numpy as np
from collections import deque
import scipy.misc as spm

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain, optimizers, Variable, serializers

class Neuralnet(Chain):

    def __init__(self, n_in, n_out):
        super(Neuralnet, self).__init__(
            L1 = F.Convolution2D(n_in, 32, ksize=8, stride=4, nobias=False, wscale=np.sqrt(2)),
            L2 = F.Convolution2D(32, 64, ksize=4, stride=2, nobias=False, wscale=np.sqrt(2)),
            L3 = F.Convolution2D(64, 64, ksize=3, stride=1, nobias=False, wscale=np.sqrt(2)),
            L4 = L.Linear(3136, 512),
            Q_value = L.Linear(512, n_out)
        )

    def Q_func(self, x):
        h = F.relu(self.L1(x/255.))
        h = F.relu(self.L2(h))
        h = F.relu(self.L3(h))
        h = F.relu(self.L4(h))
        h = self.Q_value(h)
        return h

class Agent():

    def __init__(self, n_act, seed):
        np.random.seed(seed)
        sys.setrecursionlimit(10000)

        self.n_history = 4
        self.gamma = 0.99
        self.mem_size = 1e5
        self.batch_size = 32
        self.eps = 1
        self.eps_decay = 1e-3
        self.eps_min = 0
        self.exploration = 1e3
        self.train_freq = 1
        self.target_update_freq = 1e4

        self.model = Neuralnet(self.n_history, n_act)
        self.target_model = copy.deepcopy(self.model)
        #self.optimizer = optimizers.Adam()
        self.optimizer = optimizers.RMSpropGraves(lr=0.00025, alpha=0.95, momentum=0.95, eps=0.01)
        self.optimizer.setup(self.model)

        self.n_act = n_act
        self.memory = deque()
        self.loss = 0
        self.step = 0
        self.Q = 0

    def stock_experience(self, st, act, r, st_dash, ep_end):
        self.memory.append((st, act, r, st_dash, ep_end))
        if len(self.memory) > self.mem_size:
            self.memory.popleft()

    def forward(self, st, act, r, st_dash, ep_end):
        s = Variable(st)
        s_dash = Variable(st_dash)
        Q = self.model.Q_func(s)
        tmp = self.target_model.Q_func(s_dash)
        tmp = list(map(np.max, tmp.data))
        max_Q_dash = np.asanyarray(tmp, dtype=np.float32)
        target = np.asanyarray(copy.deepcopy(Q.data), dtype=np.float32)
        for i in xrange(self.batch_size):
            target[i, act[i]] = r[i] + (self.gamma * max_Q_dash[i]) * (not ep_end[i])
        td = Variable(target) - Q  # TD error
        td_tmp = td.data + 1000.0 * (abs(td.data) <= 1)  # Avoid zero division
        td_clip = td * (abs(td.data) <= 1) + td/abs(td_tmp) * (abs(td.data) > 1)
        zero_val = Variable(np.zeros((self.batch_size, self.n_act), dtype=np.float32))
        loss = F.mean_squared_error(td_clip, zero_val)
        self.loss = loss.data
        return loss

    def suffle_memory(self):
        mem = np.array(self.memory)
        return np.random.permutation(mem)

    def parse_batch(self, batch):
        st, act, r, st_dash, ep_end = [], [], [], [], []
        for i in xrange(self.batch_size):
            st.append(batch[i][0])
            act.append(batch[i][1])
            r.append(batch[i][2])
            st_dash.append(batch[i][3])
            ep_end.append(batch[i][4])
        st = np.array(st, dtype=np.float32)
        act = np.array(act, dtype=np.int8)
        r = np.array(r, dtype=np.float32)
        st_dash = np.array(st_dash, dtype=np.float32)
        ep_end = np.array(ep_end, dtype=np.bool)
        return st, act, r, st_dash, ep_end

    def experience_replay(self):
        mem = self.suffle_memory()
        batch = mem[0:self.batch_size]
        st, act, r, st_d, ep_end = self.parse_batch(batch)
        self.model.zerograds()
        loss = self.forward(st, act, r, st_d, ep_end)
        loss.backward()
        self.optimizer.update()

    def get_action(self, st):
        if np.random.rand() < self.eps:
            return np.random.randint(0, self.n_act), 0
        else:
            s = Variable(st)
            Q = self.model.Q_func(s)
            Q = Q.data[0]
            a = np.argmax(Q)
            return np.asarray(a, dtype=np.int8), max(Q)

    def reduce_eps(self):
        if self.eps > self.eps_min and self.exploration < self.step:
            self.eps -= self.eps_decay

    def train(self):
        if len(self.memory) >= self.exploration:
            if self.step % self.train_freq == 0:
                self.experience_replay()
                self.reduce_eps()
            if self.step % self.target_update_freq == 0:
                self.target_model = copy.deepcopy(self.model)
        self.step += 1

    def scale_image(self, observation):
        img = np.dot(observation[..., :3],[0.299, 0.587, 0.114])
        return (spm.imresize(img,(110, 84)))[110-84-8:110-8,:]

    def reset_state(self, observation):
        img = self.scale_image(observation)
        self.state = np.zeros((self.n_history, 84, 84), dtype=np.uint8)
        self.state[0] = img

    def set_state(self, observation):
        img = self.scale_image(observation)
        for i in range(0, 3):
            self.state[i] = self.state[i+1].astype(np.uint8)
        self.state[3] = img.astype(np.uint8)

    def act(self):
        st = np.asanyarray(self.state.reshape(1, 4, 84, 84), dtype=np.float32)
        action, self.Q = self.get_action(st)
        return action

    def update_experience(self, observation, action, reward, ep_end):
        self.prev_state = copy.deepcopy(self.state)
        self.set_state(observation)
        st = np.asanyarray(self.prev_state.reshape(4, 84, 84), dtype=np.float32)
        st_dash = np.asanyarray(self.state.reshape(4, 84, 84), dtype=np.float32)
        self.stock_experience(st, action, reward, st_dash, ep_end)

    def save_model(self, model_dir):
        serializers.save_npz(model_dir + "model.npz", self.model)

    def load_model(self, model_dir):
        serializers.load_npz(model_dir + "model.npz", self.model)
        self.target_model = copy.deepcopy(self.model)
