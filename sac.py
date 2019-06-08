import numpy as np
import tensorflow as tf
import gym
import time
import argparse
import sys

from replay_buffer import *
from ac import *
from gym_torcs import TorcsEnv

#-----------------------------------------------------------------------------------------------

def sac_fn(args):

    sess = tf.Session()
    
    sac = SAC(sess, args)     
  
    saver = tf.train.Saver()

    sac.init_all_vars()
    sac.target_init()
    
    if (args.train_test == 0):
       print('train model ')
       train_sac(sess, args, sac, saver)
    elif (args.train_test == 1):
       print('test model ')
       test_sac(sess, args, sac, saver)
    else:
       print('wrong entry for train_test: ', args.train_test)          
       sys.exit()

#-------------------------------------------------------------------------------

def train_sac(sess, args, sac, saver):

    env = TorcsEnv(vision=False, throttle=True, gear_change=False)
    
    replay_buffer = ReplayBuffer(args.s_dim, args.a_dim, args.buff_size)

    for ep in range(args.total_ep):

        if np.mod(ep, 100) == 0:
            ob = env.reset(relaunch=True)   #relaunch TORCS every N episode because of the memory leak error
        else:
            ob = env.reset()

        s = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))


        done = False 
        ep_rew = 0.0
        ep_len = 0 

        while (not done):

            # first 10 episodes, just step on gas, drive straight
            if (ep > 10):
                a = sac.get_action(s)
            else:
                a = np.array([0.0, 1.0, 0.0])


            ob, r, done, _ = env.step(a)
            s2 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm)) 

            ep_rew += r
            ep_len += 1

            if (ep_len >= args.max_ep_len):
                 done = True

            replay_buffer.store(s, a, r, s2, float(done))

            s = s2

            batch = replay_buffer.sample_batch(args.batch_size)
            outs = sac.train(batch)

         
        print('episode: ', ep, ' | episode rewards: ', round(ep_rew,4), ' | episode length: ', ep_len, ' | alpha/temperature: ', outs[9])      
        with open("performance.txt", "a") as myfile:
              myfile.write(str(ep) + " " + str(ep_len) + " " + str(round(ep_rew,4)) + " " + str(round(outs[9],4)) + "\n")                 

        if (ep % 10 == 0):
            # save model 
            saver.save(sess, 'ckpt/model') 

#-------------------------------------------------------------------------------            

def test_sac(sess, args, sac, saver):

        saver.restore(sess, 'ckpt/model')
 
        env = TorcsEnv(vision=False, throttle=True, gear_change=False)

        ob = env.reset(relaunch=True)   #relaunch TORCS every N episode because of the memory leak error
           
        s = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
     
        done = False 
        ep_rew = 0.0
        ep_len = 0 

        while (not done):
     
                # deterministic actions at test time
                a = sac.get_action(s, True)   
                 
                ob, r, done, _ = env.step(a)
                s = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

                ep_rew += r
                ep_len += 1

                if (ep_len >= args.max_ep_len):
                    done = True

        print('test time performance: | rewards: ', ep_rew, ' | length: ', ep_len) 

#------------------------------------------------------------------------------- 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_dim', type=int, default=29)
    parser.add_argument('--a_dim', type=int, default=3)

    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--total_ep', type=int, default=2000)
    parser.add_argument('--max_ep_len', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--polyak', type=float, default=0.995)
    parser.add_argument('--buff_size', type=int, default=int(5e5))
    parser.add_argument('--batch_size', type=int, default=128)

    # train_test = 0 for train; = 1 for test
    parser.add_argument('--train_test', type=int, default=1)
    
    args = parser.parse_args()

    sac_fn(args)
