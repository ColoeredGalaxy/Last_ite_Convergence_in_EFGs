import numpy as np
from collections import defaultdict
from open_spiel.python import policy
import pyspiel

import math
from agents.omd import OMDBase
from agents.utils import sample_from_weights
from agents.ixomd import IXOMD
from agents.utils import compute_log_sum_from_logit

from open_spiel.python.algorithms.exploitability import nash_conv

from tqdm import tqdm


import random 
 


class OMD_with_Virtual_Transition_Weighted_Negentropy_Regularization(OMDBase):

  def __init__(
    self,
    game,
    budget,
    base_constant=1.0,
    lr_constant=1.0,
    lr_pow_H=-0.5,
    lr_pow_A=-0.5,
    lr_pow_X=-0.5,
    lr_pow_T=-0.5,
    lr_pow_bal=-1.0,
    ix_constant=1.0,
    ix_pow_H=-0.5,
    ix_pow_A=-0.5,
    ix_pow_X=-0.5,
    ix_pow_T=-0.5,
    ix_pow_bal=-1.0,
    T=0,
    Opt=1,
    KK=1e6,
    name=None
  ):
    OMDBase.__init__(
      self,
      game,
      budget,
      base_constant=4.6415888336127775,
      lr_constant=lr_constant,
      lr_pow_H=lr_pow_H,
      lr_pow_A=lr_pow_A,
      lr_pow_X=lr_pow_X,
      lr_pow_T=lr_pow_T,
      ix_constant=ix_constant,
      ix_pow_H=ix_pow_H,
      ix_pow_A=ix_pow_A,
      ix_pow_X=ix_pow_X,
      ix_pow_T=ix_pow_T
      )

    self.name = 'OMD_with_Virtual_Transition_Weighted_Negentropy_Regularization'
    if name:
      self.name = name

    #Balanced transitions and 
    self.T = int(float(T))
    self.Opt = Opt 
    self.KK = int(float(KK))
    self.compute_balanced()
    self.tree_structure = {}
    for h in range(1,self.game.max_game_length()+1):
      self.tree_structure[h]=[]
    for inital_state in self.initial_keys:
      self.compute_tree_structure(inital_state,1)
    self.tree_structure = dict(filter(lambda item: item[1] != [],self.tree_structure.items()))
    self.plans = np.zeros(self.policy_shape)
    for inital_state in self.initial_keys:
        self.compute_c_and_d(inital_state)
    self.initial_state_for_palyer = [[],[]]
    for i in self.initial_keys:
        self.initial_state_for_palyer[self.current_player_from_key[i]].append(i)
    self.L = np.zeros(self.policy_shape)
    self.mu_star = np.ones(self.policy_shape)
    for state_idx in range(self.policy_shape[0]):
      self.mu_star[state_idx] /= len(self.legal_actions_from_key[state_idx])
    self.current_logit=np.log(self.current_policy.action_probability_array,where=self.legal_actions_indicator)

    for inital_state in self.initial_keys:
        self.compute_virtual_transition(inital_state,True)

    self.learning_rates = self.base_learning_rate * np.ones(self.policy_shape[0])

    self.learning_rates *= self.balanced_transition_plan **lr_pow_bal

    self.implicit_explorations = self.base_implicit_exploration*np.ones(self.policy_shape)

    self.epsilion = 1e-1 
    self.thd = 1e-9


  def compute_balanced(self):
      self.initial_keys=[]
      self.depth_from_key=np.zeros(self.policy_shape[0], dtype=int)
      self.current_player_from_key = np.zeros(self.policy_shape[0], dtype=int)
      self.total_actions_from_key = np.zeros(self.policy_shape[0], dtype=int)
      self.total_actions_from_action = np.zeros(self.policy_shape)
      self.legal_actions_from_key = [[] for i in range(self.policy_shape[0])]
      self.balanced_policy=np.zeros(self.policy_shape)
      self.balanced_transition_plan=np.zeros(self.policy_shape[0],dtype=float) 
      self.key_children = [defaultdict(list) for i in range(self.policy_shape[0])] #key_children gives for each state_key a dictionnary that associates to each action the list of children 
      self.d = np.ones(self.policy_shape, dtype=int)
      self.c = np.ones(self.policy_shape[0], dtype=int)
      self.p = np.zeros(self.policy_shape[0], dtype=float)

      self.compute_information_tree_from_state(self.game.new_initial_state(),[[],[]],[0,0])
      for initial_key in self.initial_keys:
        self.compute_balanced_policy_from_key(initial_key)
        self.balanced_transition_plan[initial_key]=1.0
        for initial_action in self.legal_actions_from_key[initial_key]:
          self.compute_balanced_transition_from_action(initial_key,initial_action,1.0)

  def compute_information_tree_from_state(self, state, trajectory, depth):
    if state.is_terminal():
        return
    if state.is_chance_node():
        for action, _ in state.chance_outcomes():
            self.compute_information_tree_from_state(state.child(action), trajectory, depth)
        return
    current_player = state.current_player()
    legal_actions = state.legal_actions(current_player)
    number_legal_actions = len(legal_actions)
    state_key = self.state_index(state)
    h=depth[current_player]
    if self.total_actions_from_key[state_key] == 0:
        self.current_player_from_key[state_key] = current_player
        self.legal_actions_from_key[state_key] = legal_actions
        self.depth_from_key[state_key]=h
        
        if len(trajectory[current_player]) == 0:
          self.initial_keys.append(state_key)
        else:
          self.key_children[trajectory[current_player][-1][0]][trajectory[current_player][-1][1]].append(state_key)
        
        self.total_actions_from_key[state_key]=number_legal_actions
        for action in legal_actions:
          self.total_actions_from_action[state_key, action]=1
        for parent_couple in trajectory[current_player]:
          self.total_actions_from_key[parent_couple[0]] += number_legal_actions
          self.total_actions_from_action[parent_couple[0],parent_couple[1]]+=number_legal_actions
           
    depth[current_player]=h+1
    for action in legal_actions:
      trajectory[current_player].append([state_key,action])
      self.compute_information_tree_from_state(state.child(action), trajectory, depth)
      trajectory[current_player].pop()
    depth[current_player]=h

  
  def compute_balanced_policy_from_key(self, state_key):
    for action in self.legal_actions_from_key[state_key]:
      self.balanced_policy[state_key,action]=self.total_actions_from_action[state_key,action]/self.total_actions_from_key[state_key]
      for state_key_child in self.key_children[state_key][action]:
        self.compute_balanced_policy_from_key(state_key_child)
      
  #Compute the product of the transitions
  def compute_balanced_transition_from_action(self, state_key, action, current_transition):
    #if the current action is not the last action of the episode
    if self.total_actions_from_action[state_key,action]>1:
      for state_key_child in self.key_children[state_key][action]:
        new_transition=current_transition*self.total_actions_from_key[state_key_child]/self.total_actions_from_action[state_key,action]
        self.balanced_transition_plan[state_key_child]=new_transition
        for new_action in self.legal_actions_from_key[state_key_child] :
          self.compute_balanced_transition_from_action(state_key_child,new_action,new_transition)
    
  def compute_c_and_d(self,state_idx):
    is_terminal=True
    for action_idx in range(self.policy_shape[1]):
      if len(self.key_children[state_idx][action_idx])==0:
        self.d[state_idx][action_idx]=0
      else:
        is_terminal=False
        for childen_state_idx in self.key_children[state_idx][action_idx]:
          self.compute_c_and_d(childen_state_idx)
          self.d[state_idx][action_idx]+=self.c[childen_state_idx]
    if is_terminal==True:
      self.c[state_idx]=1
      return
    for action_idx in range(self.policy_shape[1]):
      self.c[state_idx]=max(self.c[state_idx],self.d[state_idx][action_idx])
  
  def compute_virtual_transition(self,state_idx,is_beginning):
    if is_beginning == True:
      self.p[state_idx] = self.c[state_idx] /sum([self.c[state] for state in self.initial_state_for_palyer[self.current_player_from_key[state_idx]]])

    for action_idx in range(self.policy_shape[1]):
        if len(self.key_children[state_idx][action_idx])!=0:
          for childen_state_idx in self.key_children[state_idx][action_idx]:
            self.p[childen_state_idx] = self.p[state_idx] * self.c[state_idx]/sum([self.c[state] for state in self.key_children[state_idx][action_idx]])
            self.compute_virtual_transition(childen_state_idx,False)


  def update_along_trajectory(self, trajectory,k):

    #Initialize values
    values =  np.zeros(self.num_players)

    p = np.transpose(np.tile(self.p,(self.policy_shape[1],1)))
    lr = np.transpose(np.tile(self.learning_rates,(self.policy_shape[1],1)))*(k**(-5/8))
    thd = 1e-10 
    self.current_logit += self.epsilion*k**(-1/8)*p*np.log(np.maximum(p*self.plans,thd))*lr 

    for transition in reversed(trajectory):
      player, state_idx, action_idx, plan, loss = transition.values()
      policy = self.current_policy.action_probability_array[state_idx,:]

      #Compute ix loss
      ix = self.implicit_explorations[state_idx, action_idx]*(k**(-3/8))
      ix_loss = loss/(plan+ix) 

      #Compute new policy 
      legal_actions = self.legal_actions_indicator[state_idx,:]
      lr = self.learning_rates[state_idx]
      adjusted_loss = ix_loss - values[player]
      self.current_logit[state_idx,action_idx] -= lr*adjusted_loss
      logz=compute_log_sum_from_logit(self.current_logit[state_idx,:],legal_actions)
      self.current_logit[state_idx,:] -= logz*legal_actions
      values[player] = logz/lr
      new_policy = np.exp(self.current_logit[state_idx,:],where=legal_actions)*legal_actions
 
      #Update new policy 
      self.set_current_policy(state_idx, new_policy)          



  def compute_tree_structure(self,state_idx,h):
    for action_idx in range(self.policy_shape[1]):
      if self.legal_actions_indicator[state_idx][action_idx] == 1:
        self.tree_structure[h].append(state_idx)
        for child_state in self.key_children[state_idx][action_idx]:
          self.compute_tree_structure(child_state,h+1)
    
  def fit(
    self,
    new_scale=None,
    log_interval=None,
    writer_path=None,
    record_exploitabilities = False,
    record_current = True,
    verbose=1,
    number_points=None,
    first_point=None
  ):


    self.writer_path = writer_path
    if writer_path is not None:
        self.writer = None
    else:
        self.writer = None

    #Exploitabilities returned at the end of the training
    if record_exploitabilities:
      list_exploitability = {'step':[], 'current':[], 'average':[]}
    
    if log_interval is not None:
      recorded_steps=[step for step in range(self.budget-1,0,-log_interval)]

    if number_points is not None:
      log_step=(math.log(self.budget)-math.log(first_point))/(number_points-1)
      recorded_steps=[self.budget-1]+[round(first_point*math.exp(i*log_step))-1 for i in range(number_points-2,0,-1)]+[first_point-1]
      recorded_steps= list(set(recorded_steps)) 
      recorded_steps.sort(reverse=True)

    for step in tqdm(range(self.budget), disable=(verbose < 1 )):
      #Sample a trajectory
      trajectory = self.sample_trajectory(step)
      #Update 
      self.update_along_trajectory(trajectory,step+1)

      if step==recorded_steps[-1]:
        if record_current:
          exploit_current = nash_conv(self.game, self.current_policy, use_cpp_br=True)

        self.update_average_policy()
        exploit_average = nash_conv(self.game, self.average_policy, use_cpp_br=True)
        if record_exploitabilities:
          list_exploitability['step'].append(step+1)
          if record_current:
            list_exploitability['current'].append(exploit_current)
          list_exploitability['average'].append(exploit_average)
        if self.writer:
          if record_current:
            self.writer.add_scalar('exploitability/current', exploit_current, step)
          self.writer.add_scalar('exploitability/average', exploit_average, step)
        if verbose > 1:
          tqdm.write('')
          tqdm.write(f'step: {step}')
          if record_current:
            tqdm.write(f'exploitability current: {exploit_current}')
          tqdm.write(f'exploitability average: {exploit_average}')
        recorded_steps.pop()

    if record_exploitabilities:
      list_exploitability['step'] = np.array(list_exploitability['step'])
      if record_current:
        list_exploitability['current'] = np.array(list_exploitability['current'])
      list_exploitability['average'] = np.array(list_exploitability['average'])
      return list_exploitability
