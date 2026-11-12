import torch,numpy as np
from two_sitting_ppo_env import ActorCritic as OriginalActorCritic
class ActorCritic(OriginalActorCritic):
    def centered_mean(self,x,counter):return self.actor(x)-self.actor(counter)
    def distribution(self,x,counter):
        return torch.distributions.Normal(self.centered_mean(x,counter),self.log_std.clamp(float(np.log(.001)),-.3).exp())
