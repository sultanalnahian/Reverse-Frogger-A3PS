import math
import random

import gym
from gym import wrappers
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from game_access_tonni_ppo import Game 
from torch.distributions import Normal
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import pickle

torch.manual_seed(111)
np.random.seed(111)

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, action_size):
        super(ActorCritic, self).__init__()
        #self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = 512
        self.embed_size = 512
        self.n_layers = 2
        self.conv1 = nn.Conv2d(1,16,3,stride=2)
        self.conv2 = nn.Conv2d(16,32,3,stride=2)
        self.conv3 = nn.Conv2d(32,64,3,stride=1)
        self.fc_size = 1600
        goal_state_size = 3
        # self.conv6 = nn.Conv2d(64,64,3,stride=1)
        # self.conv7 = nn.Conv2d(64,128,3,stride=1)

        # self.drop1 = nn.Dropout2d(0.3)
        # self.drop2 = nn.Dropout2d(0.3)
        # self.lstm = nn.LSTM(300,self.num_hid,batch_first=True)

        self.linear1 = nn.Linear(self.fc_size,self.embed_size)
        self.linear2_actor = nn.Linear (self.hidden_size * 2, 256)
        self.linear2_critic = nn.Linear (self.hidden_size * 2, 256)
        # self.linear3 = nn.Linear (512,128)
        self.linear_actor = nn.Linear(256 + goal_state_size,self.action_size )
        self.linear_critic = nn.Linear(256 + goal_state_size, 1)
        self.actor_lstm = nn.LSTM(self.embed_size, self.hidden_size, self.n_layers, batch_first=True, dropout=0.3)
        self.critic_lstm = nn.LSTM(self.embed_size, self.hidden_size, self.n_layers, batch_first=True, dropout=0.3)

    def init_hidden(self, batch_size):
        h0 = torch.zeros(1 * self.n_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(1 * self.n_layers, batch_size, self.hidden_size)
        
        if torch.cuda.is_available():
            h0 = h0.cuda()
            c0 = c0.cuda()
            
        return (h0, c0)

    def forward(self, state, goal_state):
        batch_size = state.shape[0]
        (h0,c0)= self.init_hidden(batch_size)
        (_h0,_c0) = self.init_hidden(batch_size)
        state = state.view(-1,1, IMAGE_WIDTH, IMAGE_HEIGHT)
        output1 = F.relu(self.conv1(state))
        output2 = F.relu(self.conv2(output1))
        output2 = F.max_pool2d(output2,2,stride=2)

        # output2 = self.drop1(output2)
        
        output3 = F.relu(self.conv3(output2))
        output3 = F.max_pool2d(output3,2,stride=2)

        # output5 = F.relu(self.conv5(output4))
        # output6 = F.relu(self.conv6(output5))
        # output7 = F.relu(self.conv7(output6))
        # output7 = F.max_pool2d(output7,2,stride=2)


        img_emb = torch.reshape(output3,(-1,STACK_SIZE, self.fc_size))
        img_emb = self.linear1(img_emb)
        
        output, (actor_h, actor_c) = self.actor_lstm(img_emb, (h0,c0))
        actor_h = actor_h.view(-1, self.hidden_size*2)
        fc_actor = self.linear2_actor(actor_h)
        fc_actor = torch.cat((fc_actor, goal_state), 1)

        output, (critic_h,critic_c) = self.critic_lstm(img_emb, (_h0,_c0))
        critic_h = critic_h.view(-1, self.hidden_size*2)
        fc_critic = self.linear2_actor(critic_h)
        fc_critic = torch.cat((fc_critic, goal_state), 1)
        
        
        dist_actor = self.linear_actor(fc_actor)
        prob_actor = F.softmax(dist_actor, dim=-1)
        dist_policy  = Categorical(prob_actor)
        # dist = Categorical(F.softmax(dist_actor, dim=-1))
        value_critic = self.linear_critic(fc_critic)               
        
        return dist_policy, value_critic

def plot(frame_idx, rewards):
    # clear_output(True)
    plt.figure(figsize=(16,8))
    # plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()
    
def test_env(vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy())
        state = next_state
        if vis: env.render()
        total_reward += reward
    return total_reward


def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        # _v0 = values[step]
        # _v1 = values[step + 1]
        # _item1 = gamma * _v1 
        # _item1 = _item1 * masks[step]
        # _item2 = _item1 - _v0
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step] *masks[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
        # returns.insert(0, gae)
    return returns

def ppo_iter(mini_batch_size, states, goal_states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        # _states = states[rand_ids, :]
        # _actions = actions[rand_ids]
        # _log_probs = log_probs[rand_ids, :]
        # _returns = returns[rand_ids, :]
        # _advantage = advantage[rand_ids, :]
        yield states[rand_ids, :], goal_states[rand_ids, :], actions[rand_ids], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
        # yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
        
        
def ppo_update(ppo_epochs, mini_batch_size, states, goal_states, actions, log_probs, returns, advantages, clip_param=0.2):
    dist, values = model(states, goal_states)
    entropy = dist.entropy().mean()
    new_log_probs = dist.log_prob(actions)
    new_log_probs = new_log_probs.unsqueeze(1)

    ratio = (new_log_probs - log_probs).exp()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages

    actor_loss  = - torch.min(surr1, surr2).mean()
    critic_loss = (returns - values).pow(2).mean()

    loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

    optimizer.zero_grad()
    loss.backward()
            # nn.utils.clip_grad_norm(model.parameters(), 40)
    optimizer.step()

    loss_log_file = open(train_log_loss_file_name, "a+")
    loss_log_file.write('%d %d %d\n' %(actor_loss, critic_loss, loss))    
    loss_log_file.close()

# def ppo_update(ppo_epochs, mini_batch_size, states, goal_states, actions, log_probs, returns, advantages, clip_param=0.2):
#     for _ in range(ppo_epochs):
#         for state, goal_state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, goal_states, actions, log_probs, returns, advantages):
#             dist, value = model(state, goal_state)
#             entropy = dist.entropy().mean()
#             new_log_probs = dist.log_prob(action)
#             new_log_probs = new_log_probs.unsqueeze(1)

#             ratio = (new_log_probs - old_log_probs).exp()
#             surr1 = ratio * advantage
#             surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

#             actor_loss  = - torch.min(surr1, surr2).mean()
#             critic_loss = (return_ - value).pow(2).mean()

#             loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

#             optimizer.zero_grad()
#             loss.backward()
#             # nn.utils.clip_grad_norm(model.parameters(), 40)
#             optimizer.step()


# num_inputs  = env.observation_space.shape[0]
# # num_outputs = env.action_space.shape[0]
# num_outputs = env.action_space.n


#Hyper params:
# hidden_size      = 256
lr               = 1e-4
num_steps        = 20
ppo_epochs       = 4
threshold_reward = -200
num_outputs = 5
STACK_SIZE = 4
mini_batch_size  = 4
IMAGE_HEIGHT = 100
IMAGE_WIDTH = 100
ENV_LOCATION = "windows_build/UnityFrogger"

model = ActorCritic(num_outputs).to(device)
model = torch.load('./frogger_model/ppo_spr_rew150000.pkl')
print("model is loaded successfully!")
optimizer = optim.Adam(model.parameters(), lr=lr)

max_frames = 250001
early_stop = False
train_log_file_name = "results/ppo_baseline_sparse_rew.txt"
train_log_loss_file_name = "results/ppo_baseline_log_loss_train.txt"

def a2c(env):
    
    frame_idx  = 150001
    action = 0
    training_rewards = []
    total_episode_reward = 0
    reward, state, done = env.perform_action(action, IMAGE_HEIGHT, IMAGE_WIDTH, STACK_SIZE)
    init_goal_state = np.array([0, 0, 0])
    num_episode = 3092
    num_moves = 0
    while frame_idx < max_frames and not early_stop:

        log_probs = []
        values    = []
        states    = []
        goal_states = []
        actions   = []
        rewards   = []
        masks     = []
        entropy = 0

        for _i_step in range(num_steps):
            num_moves +=1
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            goal_state = torch.FloatTensor(init_goal_state).unsqueeze(0).to(device)
            dist, value = model(state, goal_state)

            action = dist.sample()
            reward, next_state, done = env.perform_action(action.cpu().numpy()[0], IMAGE_HEIGHT, IMAGE_WIDTH, STACK_SIZE)
            # if done:
            #     reward = -2
            if reward == 10 and init_goal_state[0] == 0:
                init_goal_state[0] = 1
            elif reward == 100 and init_goal_state[0] == 1 and init_goal_state[1] == 0:
                init_goal_state[1] = 1
            elif reward == 400 and init_goal_state[0] == 1 and init_goal_state[1] == 1 and init_goal_state[2] ==0:
                init_goal_state[2]= 1

            print("reward: ", reward, " action: ",action.cpu().numpy()[0])
            # if reward < -1:
            #     reward = -1
            # elif reward > 1:
            #     reward = 1
            total_episode_reward += reward
            if done:
                env.reset()

                # write the stats
                train_log_file = open(train_log_file_name, "a+")
                train_log_file.write('%d %d %d\n' %(num_episode, num_moves, total_episode_reward))    
                train_log_file.close()

                training_rewards.append(total_episode_reward)

                print("episode no: ", num_episode, " episode_reward: ", total_episode_reward, " frame_id: ", frame_idx)
                total_episode_reward = 0
                num_episode +=1
                no_action = 0
                _, next_state, _ = env.perform_action(no_action, IMAGE_HEIGHT, IMAGE_WIDTH, STACK_SIZE)
                init_goal_state = np.array([0, 0, 0])
                num_moves = 0
                
            
            # next_state, reward, done, _ = env.step(action.unsqueeze(0).cpu().numpy())

            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()
            
            log_probs.append(log_prob.unsqueeze(0))
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            masks.append(torch.tensor([1-done], dtype=torch.float, device=device))
            # rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
            # masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
            
            states.append(state)
            goal_states.append(goal_state)
            actions.append(action)

            
            state = next_state
            frame_idx += 1
            
            if frame_idx % 5000 == 0:
                print("frame: ",frame_idx)
                model_path = 'frogger_model' + '/ppo_spr_rew' + str(frame_idx) + '.pkl'
                torch.save(model, model_path)
                print("model saved ..........")
            #     test_reward = np.mean([test_env(True) for _ in range(10)])
            #     test_rewards.append(test_reward)
            #     plot(frame_idx, test_rewards)
            #     if test_reward > threshold_reward: early_stop = True
                

        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        goal_state = torch.FloatTensor(init_goal_state).unsqueeze(0).to(device)
        _, next_value = model(next_state, goal_state)
        returns = compute_gae(next_value, rewards, masks, values)

        returns   = torch.cat(returns).detach()
        log_probs = torch.cat(log_probs).detach()
        values    = torch.cat(values).detach()
        states    = torch.cat(states)
        goal_states = torch.cat(goal_states)
        actions   = torch.cat(actions)
        advantage = returns - values
        
        ppo_update(ppo_epochs, mini_batch_size, states, goal_states, actions, log_probs, returns, advantage)

    pickle.dump(training_rewards, open( "results/spr_rewards_ppo.p", "wb" ) )
    fig1 = plt.figure(figsize=(12,9))
    plt.plot(training_rewards)
    plt.xlabel('no_episode', fontsize = 13)
    plt.ylabel('reward',fontsize = 13 )
    plt.show(fig1)
    

if __name__ == "__main__":
    #env = gym.make("CartPole-v0").unwrapped
    env = Game(ENV_LOCATION)
    a2c(env)