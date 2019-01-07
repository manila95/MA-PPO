"""
A simple version of OpenAI's Proximal Policy Optimization (PPO). [http://adsabs.harvard.edu/abs/2017arXiv170706347S]

Distributing workers in parallel to collect data, then stop worker's roll-out and train PPO on collected data.
Restart workers once PPO is updated.

The global PPO updating rule is adopted from DeepMind's paper (DPPO):
Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [http://adsabs.harvard.edu/abs/2017arXiv170702286H]

View more on my tutorial website: https://morvanzhou.github.io/tutorials

Dependencies:
tensorflow r1.2
gym 0.9.2
"""

import tensorflow as tf
from tensorflow.contrib.distributions import Normal, MultivariateNormalDiag
import numpy as np
#import matplotlib.pyplot as plt
import threading, queue


EP_MAX = 2000
EP_LEN = 300
N_WORKER = 1                # parallel workers
GAMMA = 0.99                 # reward discount factor
A_LR = 0.0001               # learning rate for actor
C_LR = 0.0005                # learning rate for critic
MIN_BATCH_SIZE = 256         # minimum batch size for updating PPO
UPDATE_STEP = 5             # loop update operation n-steps
EPSILON = 0.2               # Clipped surrogate objective
ETA = 1
MODE = ['easy', 'hard']
n_model = 1
from gym_torcs import TorcsEnv
import snakeoil3_gym as snakeoil3
import os
import traceback

env = TorcsEnv(vision=False, throttle=True, gear_change=False)
S_DIM = 65
A_DIM = 3 #env.action_dim
A_BOUND = [-1,1]

#import tflearn
import random
import numpy as np 

class OU(object):

    def function(self, x, mu, theta, sigma):
        return theta * (mu - x) + sigma * np.random.randn(1)

OU = OU()

class PPO(object):
    def __init__(self):
        self.sess = tf.Session()
        
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # critic
        l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
        self.v = tf.layers.dense(l1, 1)
        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.advantage = self.tfdc_r - self.v
        self.closs = tf.reduce_mean(tf.square(self.advantage))
        self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        self.sample_op = tf.squeeze(pi.sample(1), axis=0)  # choosing action
        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        #ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
        ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)
        surr = ratio * self.tfadv   # surrogate loss

        self.aloss = -tf.reduce_mean(tf.minimum(
            surr,
            tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) * self.tfadv))

        self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)
        self.saver = tf.train.Saver(max_to_keep=0)
        self.saver.restore(self.sess,"./weights_new/1070/PPO.ckpt") # weights_new/1070, 
        self.sess.run(tf.global_variables_initializer())

    def update(self):
        global GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            if GLOBAL_EP < EP_MAX:
                UPDATE_EVENT.wait()         # wait until get batch of data
                self.sess.run(self.update_oldpi_op)   # old pi to pi
                data = [QUEUE.get() for _ in range(QUEUE.qsize())]
                data = np.vstack(data)
                s, a, r = data[:, :S_DIM], data[:, S_DIM: S_DIM + A_DIM], data[:, -1:]
                adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
                [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(UPDATE_STEP)]
                [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(UPDATE_STEP)]
                UPDATE_EVENT.clear()        # updating finished
                GLOBAL_UPDATE_COUNTER = 0   # reset counter
                ROLLING_EVENT.set()         # set roll-out available

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            w_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
            l1 = tf.layers.dense(self.tfs, 200, tf.nn.relu, trainable=trainable)
            mu1 =  tf.layers.dense(l1, 1, tf.nn.tanh, trainable=trainable, kernel_initializer=w_init)
            sigma1 = tf.layers.dense(l1, 1, tf.nn.sigmoid, trainable=trainable, kernel_initializer=w_init)
            mu2 = tf.layers.dense(l1, 1, tf.nn.sigmoid, trainable=trainable)
            sigma2 = tf.layers.dense(l1, 1, tf.nn.sigmoid, trainable=trainable)
#            mu2 = tf.layers.dense(l1, 1, tf.nn.sigmoid, trainable=trainable)
#            sigma2 = tf.layers.dense(l1, 1, tf.nn.sigmoid, trainable=trainable)
            mu3 = tf.layers.dense(l1, 1, tf.nn.sigmoid, trainable=trainable, kernel_initializer=w_init)
            sigma3 = tf.layers.dense(l1, 1, tf.nn.sigmoid, trainable=trainable, kernel_initializer=w_init)
            mu = tf.concat([mu1,mu2,mu3], axis=1)
            sigma = tf.concat([sigma1, sigma2, sigma3], axis=1)
            norm_dist = MultivariateNormalDiag(loc=mu, scale_diag=sigma)
#        print([mu1,mu2,mu3].eval())
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params
        #return mu, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        b = [1,1,1]
#        print(a.shape)
        #a[2] = 0.
        a[0] = np.clip(a[0], -1, 1)
        a[1] = np.clip(a[1], 0, 1)
        a[2] = np.clip(a[2], 0, 1)
        return a

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]


Episode_reward = []
class Worker(object):
    def __init__(self, wid):
        self.wid = wid
        self.ppo = GLOBAL_PPO
        #self.i = i
        self.client = snakeoil3.Client(p=3101+self.wid, vision=False)

    def work(self):

        best = 0

        self.client.MAX_STEPS = np.inf
        self.client.get_servers_input(0)  # Get the initial input from torcs
        obs = self.client.S.d  # Get the current full-observation from torcs
        ob = env.make_observation(obs)

        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER

        while not COORD.should_stop():
            try:
               ob, self.client = env.reset(self.client)
            except Exception as e:
               print("Exception caught in reset "+str(traceback.format_exc()) )
               while True:
                try:
                  self.client = snakeoil3.Client(p=3101+self.wid, vision=False)  # Open new UDP in vtorcs
                  self.client.MAX_STEPS = np.inf
                  self.client.get_servers_input(0)  # Get the initial input from torcs
                  obs = self.client.S.d  # Get the current full-observation from torcs
                  ob = env.make_observation(obs)
                except Exception as e2:
                  print("Exception caught in reset's exception "+str(traceback.format_exc()) ) 
                else:
                  print("blahblahblah")
                  break
            ep_r = 0
            buffer_s, buffer_a, buffer_r = [], [], []

            for t in range(EP_LEN):
                if not ROLLING_EVENT.is_set():                  # while global PPO is updating
                    ROLLING_EVENT.wait()                        # wait until PPO is updated
                    buffer_s, buffer_a, buffer_r = [], [], []   # clear history buffer

                s = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm, ob.opponents)) 

                a = self.ppo.choose_action(s)
                ETA = (1.0-2*float(GLOBAL_EP/EP_MAX))
#                if self.wid == 0:
#                    ETA = -1.0
                a[0] += max(ETA, 0) * OU.function(a[0], 0.0 , 0.8, 0.40)
                a[1] += max(ETA, 0) * OU.function(a[1], 0.5, 0.80, 0.10)
                a[2] += max(ETA, 0) * OU.function(a[2], -0.1, 1.00, 0.05)
                #a[2] = 0.
                #ETA = (1-2*GLOBAL_EP/EP_MAX)
                #a[1] = abs(a[1])
                #a[2] = abs(a[2])
                
                try:
                   ob, r, done, info = env.step(t, self.client, a) 
                except Exception as e:
                   print("Exception caught in step " + str(traceback.format_exc()))
                   while True:
                     try:
                       self.client = snakeoil3.Client(p=3101+self.wid, vision=False)  # Open new UDP in vtorcs
                       self.client.MAX_STEPS = np.inf
                       self.client.get_servers_input(0)  # Get the initial input from torcs
                       obs = self.client.S.d  # Get the current full-observation from torcs
                       print(ob)
                       ob = env.make_observation(obs)
                     except Exception as e2:
                       print("Exception caught in reset's exception "+str(traceback.format_exc()) ) 
                     else:
                       print("12321321321321321321312321321321321321312321312312312312")
                       break
                   continue

                print("Episode: "+str(GLOBAL_EP)+" Step: "+str(t)+" Action: "+str(a)+"Reward: "+str(r))

                s_ = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm, ob.opponents))

                #s_, r, done = env.step(a)
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)                    # normalize reward, find to be useful

                s = s_
                ep_r += r


                GLOBAL_UPDATE_COUNTER += 1                      # count to minimum batch size
                if t == EP_LEN - 1 or GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                    v_s_ = self.ppo.get_v(s_)
                    discounted_r = []                           # compute discounted reward
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()
                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    QUEUE.put(np.hstack((bs, ba, br)))
                    if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                        ROLLING_EVENT.clear()       # stop collecting data
                        UPDATE_EVENT.set()          # globalPPO update

                    if GLOBAL_EP >= EP_MAX:         # stop training
                        COORD.request_stop()
                        break
                if done:
                  break
            if(GLOBAL_EP%10 == 0 and ep_r > best):
              best = ep_r
              ckpt_path = os.path.join('./weights_new/' + '%i'%GLOBAL_EP, 'PPO.ckpt')
              save_path = GLOBAL_PPO.saver.save(GLOBAL_PPO.sess, ckpt_path, write_meta_graph=False)
            # record reward changes, plot later
            if self.wid == 0:
                Episode_reward.append(ep_r)
                np.savetxt('episode_reward.txt', Episode_reward)
            if len(GLOBAL_RUNNING_R) == 0: GLOBAL_RUNNING_R.append(ep_r)
            else: GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1]*0.9+ep_r*0.1)
            GLOBAL_EP += 1
            print('{0:.1f}%'.format(GLOBAL_EP/EP_MAX*100), '|W%i' % self.wid,  '|Ep_r: %.2f' % ep_r, '\t |Best Ep_r: %.2f' % best ,'\t |Epsilon: %.4f' % ETA)


if __name__ == '__main__':
    GLOBAL_PPO = PPO()
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    UPDATE_EVENT.clear()    # no update now
    ROLLING_EVENT.set()     # start to roll out
    LOAD = 1
    if LOAD is 0:
     workers = [Worker(wid=i) for i in range(N_WORKER)]
      
     GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
     GLOBAL_RUNNING_R = []
     COORD = tf.train.Coordinator()
     QUEUE = queue.Queue()
     threads = []

    #i=0
     for worker in workers:  # worker threads
        #client = snakeoil3.Client(p=3101+i, vision=False)  # Open new UDP in vtorcs
        #client.MAX_STEPS = np.inf
        #i+=1
        t = threading.Thread(target=worker.work, args=())
        t.start()
        threads.append(t)
    # add a PPO updating thread
     threads.append(threading.Thread(target=GLOBAL_PPO.update,))
     threads[-1].start()
     COORD.join(threads)
     #save_path = saver.save(sess, "./model.ckpt")
    else:
     # plot reward change and testing
     #plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
     #plt.xlabel('Episode'); plt.ylabel('Moving reward'); plt.ion(); plt.show()
     #env.set_fps(30)
     while True:
        client = snakeoil3.Client(p=3101, vision=False)  
        ob, client = env.reset(client)
        for t in range(4000):
            s = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm, ob.opponents)) 
            a = GLOBAL_PPO.choose_action(s)
            #a[1] = abs(a[1])
            #a[2] = abs(a[2])
            #env.render()
            ob, r, done, info = env.step(t,client, a)
            print("reward at current step " + str(r)) 

