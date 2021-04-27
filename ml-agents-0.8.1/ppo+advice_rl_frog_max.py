import math
import random

import gym
from gym import wrappers
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from game_access_t import Game 
from torch.distributions import Normal
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import pickle
#from advice_generator.advice_models import EncoderCNN, DecoderRNN
from advice_support_codes.build_vocab_v2 import Vocabulary
from advice_support_codes.advice_models import EncoderCNN, DecoderRNN, Advice_Encoder, Action_Decoder
from advice_support_codes.pytorch_img_cap_models import Encoder, DecoderWithAttention

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
        #dist_policy  = Categorical(prob_actor)
        # dist = Categorical(F.softmax(dist_actor, dim=-1))
        value_critic = self.linear_critic(fc_critic)               
        
        return prob_actor, value_critic

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
    dist_policy, values = model(states, goal_states)
    dist = Categorical(dist_policy)
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
image_height1 = 100
image_width1 = 100
ENV_LOCATION = "windows_build/UnityFrogger"

model = ActorCritic(num_outputs).to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)

max_frames = 150001
early_stop = False
#train_log_file_name = "results/ppo_baseline_log_train.txt"
train_log_loss_file_name = "results/ppo_baseline_log_loss_train.txt"

def a2c(env, vocab, advice_gen_img_encoder, advice_gen_decoder, advice_driven_img_encoder, advice_driven_encoder, action_driven_decoder, train_log_file):
    
    frame_idx  = 0
    action = 0
    alpha = 0.5
    training_rewards = []
    total_episode_reward = 0
    reward, state, adv_state, done = env.perform_action(action, IMAGE_HEIGHT, IMAGE_WIDTH,image_height1, image_width1, STACK_SIZE)
    init_goal_state = np.array([0, 0, 0])
    num_episode = 0
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
            adv_state = torch.FloatTensor(adv_state).to(device)
            goal_state = torch.FloatTensor(init_goal_state).unsqueeze(0).to(device)
            dist, value = model(state, goal_state)

            with torch.no_grad():
                number_of_frames = len(adv_state)
                state_img = adv_state[number_of_frames-1]
                state_img = state_img.unsqueeze(0).to(device)
                image_feature = advice_gen_img_encoder(state_img)                 
                advice_ids = advice_gen_decoder.inference(image_feature, vocab)
                advice_ids = torch.tensor(advice_ids).to(device)
                advice_ids = advice_ids.view(1,-1)
                    # embedded_word = word_emb(advice_ids)
                img_features = advice_driven_img_encoder(state_img)
                embedded_adv,hidden = advice_driven_encoder(advice_ids)                
                action_dist = action_driven_decoder(img_features, hidden)
                action_dist_prob = F.softmax(action_dist, dim=-1)
            
            #a2c_dist_prob = F.softmax(dist, dim=-1)
                # a2c_categorical_prob = Categorical(a2c_dist_prob)
                # _test_action = a2c_categorical_prob.sample()
            combined_prob_dist = (alpha * dist) + ((1 - alpha) * action_dist_prob.detach())
            combined_prob = F.softmax(combined_prob_dist, dim=-1)
            action_prob = Categorical(combined_prob)
            action= torch.argmax(combined_prob,dim=1)

            #action = action_prob.sample()
            #action = dist.sample()

            reward, next_state, next_adv_state, done = env.perform_action(action.cpu().numpy()[0], IMAGE_HEIGHT, IMAGE_WIDTH,image_height1, image_width1, STACK_SIZE)
            # if done:
            #     reward = -2
            if reward == 50 and init_goal_state[0] == 0:
                init_goal_state[0] = 1
            elif reward == 50 and init_goal_state[0] == 1 and init_goal_state[1] == 0:
                init_goal_state[1] = 1
            elif reward == 100 and init_goal_state[0] == 1 and init_goal_state[1] == 1 and init_goal_state[2] ==0:
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
                _, next_state, next_adv_state, _ = env.perform_action(no_action, IMAGE_HEIGHT, IMAGE_WIDTH, image_height1, image_width1, STACK_SIZE)
                init_goal_state = np.array([0, 0, 0])
                num_moves = 0
                            
            # next_state, reward, done, _ = env.step(action.unsqueeze(0).cpu().numpy())
            log_prob = action_prob.log_prob(action)
            #combined_prob = combined_prob.squeeze(0)
            #log_prob = torch.log(combined_prob[action])
            entropy = action_prob.entropy().mean()
            #entropy += combined_prob.entropy().mean()
            
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
            adv_state = next_adv_state
            frame_idx += 1
            
            if frame_idx % 5000 == 0:
                print("frame: ",frame_idx)
                model_path = 'frogger_model' + '/ppo+adv_gen_git_repo_max_' + str(frame_idx) + '.pkl'
                torch.save(model, model_path)
                print("model saved ..........")
            #     test_reward = np.mean([test_env(True) for _ in range(10)])
            #     test_rewards.append(test_reward)
            #     plot(frame_idx, test_rewards)
            #     if test_reward > threshold_reward: early_stop = True

            if frame_idx % 50000 == 0:
                alpha = alpha + 0.1
                

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

    pickle.dump(training_rewards, open( "results/rewards_ppo+adv_argmax_gen_git_repo.p", "wb" ) )
    fig1 = plt.figure(figsize=(12,9))
    plt.plot(training_rewards)
    plt.xlabel('no_episode', fontsize = 13)
    plt.ylabel('reward',fontsize = 13 )
    plt.show(fig1)
    

if __name__ == "__main__":
    #env = gym.make("CartPole-v0").unwrapped
    parser = argparse.ArgumentParser()
    n_layer = 1

    parser.add_argument('--img_embed_size', type=int , default=2048,
                        help='dimension of image embedding')
    parser.add_argument('--advice_rnn_hidden_size', type=int , default=512,
                        help='dimension of advice generator lstm hidden states')                        
    parser.add_argument('--vocab_size', type=int , default=539,
                        help='Number of words in the vocabulary')
    parser.add_argument('--word_embed_size', type=int , default=300,
                        help='dimension of word embedding vectors')
    parser.add_argument('--sent_encoder_input_dim', type=int, default=256,
                        help='input dimension of sentence encoder')
    parser.add_argument('--sent_encoder_hidden_dim', type=int , default=512,
                        help='dimension of sentence encoder lstm hidden state')
    parser.add_argument('--num_output', type=int , default=5,
                        help='dimension of output')
    parser.add_argument('--sent_emb_size', type=int , default=512,
                        help='dimension of sentence embedding')
    parser.add_argument('--attention_dim', type=int, default=512,
                        help='dimension of advice gen decoder')
    
    parser.add_argument('--advice_gen_encoder_path', type=str , default='./train_advice_agent/encoder_no_stpw_end_pretrain_we.pth',
                        help='pretrained model of advice encoder') #encoder_prep95_dl-300.pth(our adv_gen)
    parser.add_argument('--advice_gen_decoder_path', type=str , default='./train_advice_agent/decoder_no_stpw_end_pretrain_we.pth',
                        help='pretrained model of advice decoder')#decoder_prep95_dl-300.pth(our adv_gen)

    parser.add_argument('--advice_driven_img_encoder_path', type=str , default='./train_advice_agent/img_encoder_retrain_pp_nm_pwe-0.pth',
                        help='pretrained model of action image encoder') #image_encoder_retrain_pp95-5.pth(our_retrained_adv_agent)
    parser.add_argument('--word_emb_path', type=str , default='./train_advice_agent/word_emb-100.pth',
                        help='pretrained model of word embedding')
    parser.add_argument('--advice_driven_encoder_path', type=str , default='./train_advice_agent/adv_encoder_retrain_pp_nm_pwe-0.pth',
                        help='pretrained model of advice encoder')#advice_encoder_retrain_pp95-5.pth(our_retrained_adv_agent)
    parser.add_argument('--advice_driven_decoder_path', type=str , default='./train_advice_agent/act_decoder_retrain_pp_nm_pwe-0.pth',
                        help='pretrained model of action decoder')#action_decoder_retrain_pp95-5.pth


    parser.add_argument('--hidden_size', type=int , default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_iterations', type=int, default=15000)
    parser.add_argument('--batch_size', type=int, default=8)
    
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    args = parser.parse_args()

    #advice_gen_img_encoder = EncoderCNN(args.img_embed_size)
    #advice_gen_decoder = DecoderRNN(args.word_embed_size, args.img_embed_size, args.advice_rnn_hidden_size, args.vocab_size)
    advice_gen_img_encoder = Encoder() 
    advice_gen_decoder = DecoderWithAttention(args.attention_dim, args.word_embed_size, args.advice_rnn_hidden_size, args.vocab_size, args.img_embed_size, 0.5)

    advice_driven_img_encoder = EncoderCNN(args.img_embed_size)
    advice_driven_encoder = Advice_Encoder(
        args.vocab_size, args.word_embed_size, args.advice_rnn_hidden_size, n_layer, 0.0)
    action_driven_decoder = Action_Decoder(args.img_embed_size, args.sent_emb_size, args.num_output, 0.2)

    if torch.cuda.is_available():
        advice_gen_img_encoder.cuda()
        #desc_encoder.cuda()
        advice_gen_decoder.cuda()
        advice_driven_img_encoder.cuda()
        # word_emb.cuda()
        advice_driven_encoder.cuda()
        action_driven_decoder.cuda()
        print("Cuda is enabled...")

    advice_gen_img_encoder.load_state_dict(torch.load(args.advice_gen_encoder_path))
    advice_gen_decoder.load_state_dict(torch.load(args.advice_gen_decoder_path))

    advice_driven_img_encoder.load_state_dict(torch.load(args.advice_driven_img_encoder_path))
    # word_emb.load_state_dict(torch.load(args.word_emb_path))
    advice_driven_encoder.load_state_dict(torch.load(args.advice_driven_encoder_path))
    action_driven_decoder.load_state_dict(torch.load(args.advice_driven_decoder_path))


    train_log_file_name = "results/ppo+adv_max_train_rew_clip_gen_git_repo.txt"
    train_log_file = open(train_log_file_name, "w")
    train_log_file.write('episode_no loss number_of_moves total_episode_reward\n')
    train_log_file.close()

    # test_log_file_name = "results/ppo+adv_test_stat_rew_clip.txt"
    # test_log_file = open(test_log_file_name, "w")
    # test_log_file.write('episode avg_moves avg_reward\n')
    # test_log_file.close()

    with open("advice_generator/vocab_frogger_preprocessed.pkl", 'rb') as f:
        vocab = pickle.load(f)


    env = Game(ENV_LOCATION)
    a2c(env, vocab, advice_gen_img_encoder, advice_gen_decoder, advice_driven_img_encoder, advice_driven_encoder, action_driven_decoder, train_log_file)