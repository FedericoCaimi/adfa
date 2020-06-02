import gym
import numpy as np
from os import path as ospath
import pickle
import matplotlib.pyplot as plt


VAR_UNO_SPACE = np.linspace(-2,2,8)#se toma de la documentacion
VAR_DOS_SPACE = np.linspace(-1,2,6)#se toman los valores en base a los maximos y minimos obtenidos
VAR_TRES_SPACE = np.linspace(-2,2,8)#en la practica esta en radianes
VAR_CUATRO_SPACE = np.linspace(-2,1,6)#se toman los valores en base a los maximos y minimos obtenidos
VAR_CINCO_SPACE = np.linspace(-3,3,12)
VAR_SEIS_SPACE = np.linspace(-7,7,14)
VAR_SIETE_SPACE = np.linspace(0,1,2)
VAR_OCHO_SPACE = np.linspace(0,1,2)
class CartPoleAgent():

    def __init__(self, env: gym.Env, path, load_q): 
        self.env = env
        self.ACTIONS = list(range(self.env.action_space.n))
        #self.ACTIONS = self.env.action_space
        print('ACTIONS: ',self.env.action_space, self.env.action_space.n, self.ACTIONS)
        self.path = path
        self.initializate_Q()
        if load_q and ospath.exists(self.path):
            with(open(self.path,'rb')) as pickle_file:
                self.Q = pickle.load(pickle_file)

    def initializate_Q(self):
        states = []

        for var_uno in range(len(VAR_UNO_SPACE)+1):
            for var_dos in range(len(VAR_DOS_SPACE)+1):
                for var_tres in range(len(VAR_TRES_SPACE)+1):
                    for var_cuatro in range(len(VAR_CUATRO_SPACE)+1):
                        for var_cinco in range(len(VAR_CINCO_SPACE)+1):
                            for var_seis in range(len(VAR_SEIS_SPACE)+1):
                                for var_siete in range(len(VAR_SIETE_SPACE)+1):
                                    for var_ocho in range(len(VAR_OCHO_SPACE)+1):
                                        states.append((var_uno,var_dos,var_tres,var_cuatro,var_cinco,var_seis,var_siete,var_ocho))
        self.Q = dict()
        for state in states:
            for action in self.ACTIONS:
                self.Q[state,action] = 0

    def get_state(self,obs):
        var_uno,var_dos,var_tres,var_cuatro,var_cinco,var_seis,var_siete,var_ocho = obs
        #print('obs: ',obs)
        #print('var_uno: ',var_uno)
        #print('VAR_UNO_SPACE:', VAR_UNO_SPACE)
        var_uno_bin = int(np.digitize(var_uno,VAR_UNO_SPACE))
        #print('var_uno_bin: ',var_uno_bin)

        #print('var_dos: ',var_dos)
        #print('VAR_DOS_SPACE:', VAR_DOS_SPACE)
        var_dos_bin = int(np.digitize(var_dos,VAR_DOS_SPACE))
        #print('var_dos_bin: ',var_dos_bin)

        #print('var_tres: ',var_tres)
        #print('VAR_TRES_SPACE:', VAR_TRES_SPACE)
        var_tres_bin = int(np.digitize(var_tres,VAR_TRES_SPACE))
        #print('var_tres_bin: ',var_tres_bin)

        #print('var_cuatro: ',var_cuatro)
        #print('VAR_CUATRO_SPACE:', VAR_CUATRO_SPACE)
        var_cuatro_bin = int(np.digitize(var_cuatro,VAR_CUATRO_SPACE))
        #print('var_cuatro_bin: ',var_cuatro_bin)

        #print('var_cinco: ',var_cinco)
        #print('VAR_CINCO_SPACE:', VAR_CINCO_SPACE)
        var_cinco_bin = int(np.digitize(var_cinco,VAR_CINCO_SPACE))
        #print('var_cinco_bin: ',var_cinco_bin)

        #print('var_cinco: ',var_seis)
        #print('VAR_CINCO_SPACE:', VAR_SEIS_SPACE)
        var_seis_bin = int(np.digitize(var_seis,VAR_SEIS_SPACE))
        #print('var_seis_bin: ',var_seis_bin)

        #print('var_cinco: ',var_siete)
        #print('VAR_CINCO_SPACE:', VAR_SIETE_SPACE)
        var_siete_bin = int(np.digitize(var_siete,VAR_SIETE_SPACE))
        #print('var_cuatro_bin: ',var_siete_bin)

        #print('var_cinco: ',var_ocho)
        #print('VAR_OCHO_SPACE:', VAR_OCHO_SPACE)
        var_ocho_bin = int(np.digitize(var_ocho,VAR_OCHO_SPACE))
        #print('var_cuatro_bin: ',var_ocho_bin)

        return var_uno_bin,var_dos_bin,var_tres_bin,var_cuatro_bin,var_cinco_bin,var_seis_bin,var_siete_bin,var_ocho_bin

    #eps (probablilidad de explorar)
    #alpha (learning rate, cuanto confio en el error)
    #gamma (cuan importante es la recompensa a futuro)
    def learn(self,episodes=50000, max_steps=4000 ,eps=0.9, alpha=0.9, gamma=0.99):
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
                eps = self.new_percent_less(eps,1)
                print('eps:', eps)
                #disminuyo alpha
                alpha = self.new_percent_less(alpha,1)
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
        #print('Q: ',self.Q)
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
            #self.env.render()
            #tomo una accion aleatoria del ambiente
            action = self.ACTIONS.sample()
            #print('action: ',action)
            obs, reward, done, info = self.env.step(action)
            all_obs.append(obs)
            #print('obs: ',obs)
            #print('obs: ',obs)
            if (done):
                self.env.reset() 
        print('min:', np.min(all_obs,axis=0))
        print('max:',np.max(all_obs,axis=0))
        self.env.close() 

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    learn = True

    agent = CartPoleAgent(env, path = 'LunarLander.pkl', load_q = not learn)

    if learn:
        agent.learn()
    agent.play_Q(max_steps=2000)
    #agent.play_random()
