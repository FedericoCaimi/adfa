import gym
import numpy as np
from os import path as ospath
import pickle
import matplotlib.pyplot as plt

CART_POS_SPACE = np.linspace(-4.8,4.8,20)#se toma de la documentacion
CART_VEL_SPACE = np.linspace(-3,3,24)#se toman los valores en base a los maximos y minimos obtenidos
POLE_ANG_SPACE = np.linspace(-0.42,0.42,42)#en la practica esta en radianes
POLE_VEL_SPACE = np.linspace(-3.5,3,24)#se toman los valores en base a los maximos y minimos obtenidos
class CartPoleAgent():

    def __init__(self, env: gym.Env, path, load_q): 
        self.env = env
        self.ACTIONS = list(range(self.env.action_space.n))
        print('ACTIONS: ',self.env.action_space, self.env.action_space.n, self.ACTIONS)
        self.path = path
        self.initializate_Q()
        if load_q and ospath.exists(self.path):
            with(open(self.path,'rb')) as pickle_file:
                self.Q = pickle.load(pickle_file)

    def initializate_Q(self):
        states = []

        for car_pos in range(len(CART_POS_SPACE)+1):
            for car_vel in range(len(CART_VEL_SPACE)+1):
                for pole_ang in range(len(POLE_ANG_SPACE)+1):
                    for pole_vel in range(len(POLE_VEL_SPACE)+1):
                        states.append((car_pos,car_vel,pole_ang,pole_vel))
        self.Q = dict()
        for state in states:
            for action in self.ACTIONS:
                self.Q[state,action] = 0

    def get_state(self,obs):
        posC,velC,angP,velP = obs
        posC_bin = int(np.digitize(posC,CART_POS_SPACE))
        velC_bin = int(np.digitize(velC,CART_VEL_SPACE))
        angP_bin = int(np.digitize(angP,POLE_ANG_SPACE))
        velP_bin = int(np.digitize(velP,POLE_VEL_SPACE))
        return posC_bin,velC_bin,angP_bin,velP_bin

    #eps (probablilidad de explorar)
    #alpha (learning rate, cuanto confio en el error)
    #gamma (cuan importante es la recompensa a futuro)
    def learn(self,episodes=10000, max_steps=4000 ,eps=0.9, alpha=0.9, gamma=0.99):
        self.env._max_episode_steps = max_steps

        total_rewards = []
        for i in range(0,episodes):
            done = False
            obs = self.env.reset()
            state = self.get_state(obs)
            episode_reward = 0
            while not done:
                action = self.epsilon_greedy_action(state,eps)
                obs, reward, done, _ = self.env.step(action)
                episode_reward += reward
                next_state = self.get_state(obs)
                _, value_next_action = self.max_action(next_state)
                self.Q[state, action] = self.Q[state, action] + (alpha * (reward + (gamma * value_next_action) - self.Q[state, action]))
                state = next_state
                if i % 100 == 0 and i > 0:
                    self.env.render()

            total_rewards.append(episode_reward)
            if i % 100 == 0 and i > 0:
                print('Episode: ', i, 'episode_reward: ', episode_reward)
                self.env.close()
                #disminuyo epsilo
                eps = self.new_percent_less(eps,3)
                print('eps:', eps)
                #disminuyo alpha
                alpha = self.new_percent_less(alpha,3)
                print('alpha:', alpha)

        with(open(self.path,'wb')) as pickle_file:
            pickle.dump(self.Q, pickle_file)

        mean_50_episodes_rewards = []
        for t in range(episodes):
            mean_50_episodes_rewards.append(np.mean(total_rewards[max(0,t-50):(t+1)]))
        plt.plot(mean_50_episodes_rewards)
        plt.show()

    def new_percent_less(self,val,percent):
        num = (percent * val) / 100.0
        return val-num

    def max_action(self, next_state):
        index_of_max_action = np.argmax([self.Q[next_state, a] for a in self.ACTIONS])
        return self.ACTIONS[index_of_max_action], self.Q[next_state,index_of_max_action]

    def epsilon_greedy_action(self, state, eps):
        best_action, _ = self.max_action(state)
        return np.random.choice(self.ACTIONS) if np.random.random() <= eps else best_action

    def play_Q(self, max_steps=200):
        self.env._max_episode_steps = max_steps
        obs = self.env.reset()
        state = self.get_state(obs)
        total_reward = 0
        for step in range(max_steps):
            self.env.render()
            action, _ = self.max_action(state)
            obs, reward, done, _ = self.env.step(action)
            state = self.get_state(obs)
            print('step:', step, 'state:', state,  'action:', action, 'reward:',reward, 'done:', done)
            total_reward += reward
            if done:
                break
        self.env.close()
        print('tota reward:', total_reward)

    def play_random(self,max_steps=2000):
        self.env.reset()
        all_obs = []
        for step in range(0,max_steps):
            self.env.render()
            #tomo una accion aleatoria del ambiente
            action = self.ACTIONS.sample()
            obs, reward, done, info = self.env.step(action)
            all_obs.append(obs)
            if (done):
                self.env.reset() 
        print(np.max(all_obs,axis=0),np.min(all_obs,axis=0))
        self.env.close() 

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    learn = False

    agent = CartPoleAgent(env, path = 'CartPole.pkl', load_q = not learn)

    if learn:
        agent.learn()
    agent.play_Q(max_steps=2000)
    #agent.play_random()
