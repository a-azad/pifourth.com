import gym
import time
import numpy as np
import pandas as pd
from lib.ddqn import Solver
from collections import deque
import matplotlib.pyplot as plt


class LunarLanderAgent:

    def __init__(self, gamma_val, learning_rate):
        seed_val = 23405
        self.scores = []
        self.max_itr = []
        self.env = gym.make('LunarLander-v2')
        self.env.seed(seed_val)
        self.agent = Solver(state_size=8, action_size=4,
                            seed=seed_val,
                            buffer_size=int(1e5), batch_size=100, gamma_val=gamma_val,
                            update_freq=4, learning_rate=learning_rate, tau_val=1e-3
                            )

    def machine_teaching(self, max_allowed_episodes, max_allowed_iteration,
                         epsilon_zero, epsilon_one, epsilon_rate_of_decay):
        start_time = time.time()
        scores_window = deque(maxlen=100)
        eps = epsilon_zero
        episode_counter = 0
        avg_reward = 0

        while (episode_counter < max_allowed_episodes) and (avg_reward < 200.0):
            episode_counter += 1
            state = self.env.reset()
            score = 0
            itr_counter = 0
            done = False
            while itr_counter < max_allowed_iteration and not done:
                itr_counter += 1
                action = self.agent.act(state, eps)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward

            scores_window.append(score)
            self.scores.append(score)
            self.max_itr.append(itr_counter)
            eps = max(epsilon_one, epsilon_rate_of_decay * eps)
            avg_reward = np.mean(scores_window)
            print(f'\r---Episode {episode_counter}\tAverage 100-Rewards: {avg_reward:.2f}', end="")
            if episode_counter % 100 == 0:
                print(f'\r---Episode {episode_counter}\tAverage 100-Reward: {avg_reward:.2f}')
        print(f'\n---Game trained for {episode_counter:d} episodes!')
        return time.time() - start_time, avg_reward

    def viz_results(self, trn_results_tag):
        plt.close()
        plt.plot(np.arange(len(self.max_itr)), self.max_itr, color='tab:olive')
        plt.title(f'Training: {trn_results_tag}')
        plt.ylabel('Number of Iteration Before Episode Ends')
        plt.xlabel('Episode Number')
        plt.savefig(f'./results/training_itr_{trn_results_tag}.png')

        plt.close()
        plt.plot(np.arange(len(self.scores)), self.scores, color='tab:cyan')
        plt.title(f'Training: {trn_results_tag}')
        plt.ylabel('Cumulative Episode Reward')
        plt.xlabel('Episode Number')
        plt.savefig(f'./results/training_reward_{trn_results_tag}.png')

    def test_learner(self, n_ep, test_results_tag, do_render=False):
        plt.close()
        reward_collection = []
        for _ in range(n_ep):
            total_episode_reward = 0
            done = False
            state = self.env.reset()
            while not done:
                action = self.agent.act(state)
                if do_render:
                    self.env.render()
                state, reward, done, _ = self.env.step(action)
                total_episode_reward += reward
            reward_collection.append(total_episode_reward)
        self.env.close()
        plt.plot(np.arange(len(reward_collection)), reward_collection, color='tab:blue')
        plt.title(f'Test: {test_results_tag}')
        plt.ylabel('Cumulative Episode Reward')
        plt.xlabel('Episode Number')
        plt.savefig(f'./results/test_reward_{test_results_tag}.png')
        return sum(reward_collection) / len(reward_collection)


def main(n_eps, n_itr,
         eps0, eps1, eps_decay,
         gamma_val, learning_rate,
         n_ep, trn_results_tag, test_results_tag, do_render):

    agent = LunarLanderAgent(gamma_val=gamma_val, learning_rate=learning_rate)
    exec_time_counter, trn_reward = agent.machine_teaching(max_allowed_episodes=n_eps,
                                                           max_allowed_iteration=n_itr,
                                                           epsilon_zero=eps0,
                                                           epsilon_one=eps1,
                                                           epsilon_rate_of_decay=eps_decay)
    agent.viz_results(trn_results_tag)
    test_reward = agent.test_learner(n_ep, test_results_tag, do_render)

    return exec_time_counter, trn_reward, test_reward


if __name__ == "__main__":

    all_results = []
    for i in [10, 100, 1000]:
        for g in [0.3, 0.6, 0.99]:
            for r in [1e-3, 1e-4, 1e-5]:
                for ed in [0.8, 0.9, 0.995]:
                    the_tag = f"ItrCap:{i}, Gamma:{g}, LearnRate:{r}, EpsilonDecay:{ed}"
                    print(the_tag)
                    t, r_trn, r_test = main(n_eps=1000, n_itr=1000,
                                            eps0=1.0, eps1=0.01, eps_decay=ed,
                                            gamma_val=0.99, learning_rate=5e-4,
                                            n_ep=100, trn_results_tag=the_tag, test_results_tag=the_tag,
                                            do_render=False)
                    all_results.append([i, g, r, ed, r_trn, r_test, t])

                    # print(f'---Operation completed in {t // 60:.0f} min and {t % 60:.0f} sec')
                    # print(f'---Last reward in training: {r_trn:.2f},  Average reward in test: {r_test:.2f}')

    pd.DataFrame(all_results, columns=['IterationCap', 'Gamma', 'LearningRate', 'EpsilonDecay',
                                       'FinalTrainingReward', 'AvgTestAward',
                                       'ComputeTime(s)']).to_csv('summary.csv', index=False)