from DDPG import DDPG
from SAC import SAC
import gym 

env = gym.make('CartPole-v0')
obs,rw,done, info = env.reset()
print(env.action_space.)
print(env.action_space.high)
print(env.obs_space.low)
print(env.obs_space.high)
print(obs)
print(rw)
print(done)
print(info)



def main():
    ddpg = DDPG(10, 10e-3, 10e-3, 10e-3)
    print(ddpg.env,ddpg.epoch,ddpg.learning_rate)
   
    # sac = SAC(10, 10e-3, 10e-3, 10e-3)    
    model, env = ddpg.train("Carpole-V0")
    # model, env = sac.train("Hopper_V0")    
    # no_test = 10
    # avg_return = ddpg.test(no_test, model, env)
    # print("Test score %0.2f" % (avg_return))


if __name__ == "__main__":
    main()
