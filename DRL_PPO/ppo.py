import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, actor_layers, bidirectional, actor_dropout=0.0):
        super().__init__()
        nums = 2 if bidirectional else 1
        self.LSTM = nn.LSTM(input_size=state_dim,
                            hidden_size=hidden_dim,
                            num_layers=actor_layers,
                            bias=True,
                            batch_first=True,
                            dropout=actor_dropout,
                            bidirectional=bidirectional)
        for name, param in self.LSTM.named_parameters():
            nn.init.uniform_(param,-0.1,0.1)

        self.linear = nn.Sequential(
            nn.Linear(hidden_dim * nums, hidden_dim * nums),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim * nums, action_dim)
        )
        self.apply(self.weight_init)

    def forward(self, x):
        output, (_, _) = self.LSTM(x)
        return self.linear(output)
    
    @staticmethod
    def weight_init(m):
        if isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias,0)
        elif isinstance(m,nn.BatchNorm1d):
            nn.init.constant_(m.weight,1)
            nn.init.constant_(m.bias,0)

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim, critic_layers, bidirectional, critic_dropout=0.0):
        super().__init__()
        nums = 2 if bidirectional else 1
        self.LSTM = nn.LSTM(input_size=state_dim,
                            hidden_size=hidden_dim,
                            num_layers=critic_layers,
                            bias=True,
                            batch_first=True,
                            dropout=critic_dropout,
                            bidirectional=bidirectional)
        for name, param in self.LSTM.named_parameters():
            nn.init.uniform_(param,-0.1,0.1)
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim * nums, hidden_dim * nums),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim * nums, 1)
        )
        self.apply(self.weight_init)

    def forward(self, x):
        output, (_, _) = self.LSTM(x)
        return self.linear(output)
    
    @staticmethod
    def weight_init(m):
        if isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias,0)
        elif isinstance(m,nn.BatchNorm1d):
            nn.init.constant_(m.weight,1)
            nn.init.constant_(m.bias,0)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, actor_layers, actor_dropout, actor_bidirectional, critic_layers, critic_dropout, critic_bidirectional, action_std_init, device):
        super(ActorCritic, self).__init__()
        self.device = device
        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        self.actor = Actor(state_dim, hidden_dim, action_dim, actor_layers, actor_bidirectional, actor_dropout)
        self.critic = Critic(state_dim, hidden_dim, critic_layers, critic_bidirectional, critic_dropout)

        self.apply(self._init_weights)  

    def forward(self):
        pass
    
    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)

    def action(self, state):
        action_mean = self.actor(state)  
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()

        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)
        
        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class PPO(nn.Module):
    def __init__(self, config):
        super(PPO, self).__init__()

        self.config = config
        self.gamma = config.gamma
        self.eps_clip = config.eps_clip
        self.K_epochs = config.K_epochs
        self.action_std = config.action_std_init
        self.device = config.device

        self.buffer = RolloutBuffer()

        # 定义两个网络
        self.policy = ActorCritic(
            state_dim = config.state_dim, 
            action_dim = config.action_dim, 
            hidden_dim = config.hidden_dim,
            actor_layers = config.actor_layers,
            actor_dropout = config.actor_dropout,
            actor_bidirectional = config.actor_bidirectional,
            critic_layers = config.critic_layers,
            critic_dropout =  config.critic_dropout,
            critic_bidirectional = config.critic_bidirectional,
            action_std_init = config.action_std_init, 
            device = config.device
        ).to(config.device)

        self.optimizer_actor = torch.optim.Adam(self.policy.actor.parameters(), lr=config.lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.policy.critic.parameters(), lr=config.lr_critic)
        
        self.policy_old = ActorCritic(
            state_dim = config.state_dim, 
            action_dim = config.action_dim, 
            hidden_dim = config.hidden_dim,
            actor_layers = config.actor_layers,
            actor_dropout = config.actor_dropout,
            actor_bidirectional = config.actor_bidirectional,
            critic_layers = config.critic_layers,
            critic_dropout =  config.critic_dropout,
            critic_bidirectional = config.critic_bidirectional,
            action_std_init = config.action_std_init, 
            device = config.device
        ).to(config.device)

        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if (self.action_std <= min_action_std):
            self.action_std = min_action_std
        self.set_action_std(self.action_std)


    def select_action(self, state):
        with torch.no_grad():
            state = state.to(self.device)
            action, action_logprob, state_val = self.policy_old.action(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.detach().cpu()

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.stack(rewards, dim=1).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=1)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=1)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=1)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=1)).detach().to(self.device)

        advantages = rewards.detach() - old_state_values.detach()

        for _ in range(self.K_epochs):

            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)

            ratios = torch.exp(logprobs - old_logprobs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss_critic = self.MseLoss(state_values, rewards)
            self.optimizer_critic.zero_grad()
            loss_critic.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.config.max_norm)
            self.optimizer_critic.step()

            loss_actor = -torch.min(surr1, surr2) - 0.05 * dist_entropy
            self.optimizer_actor.zero_grad()
            loss_actor.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.config.max_norm)
            self.optimizer_actor.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))