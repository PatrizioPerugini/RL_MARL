from gym_derk.envs import DerkEnv
import numpy as np
import random


#action n is a [num_agent,num_action] matrix,
#each row corresponds to the action taken by that agent and it's a list of dim 5, where:
#[MoveX (-1,1) , Rotate (-1,1), ChaseFocus(0,1), CastingSlot(int in {0..3}), Change_focus(int in {0...7})]


class Discrete_actions_space():#DerkEnv.action_space):
    '''
    def __init__(self,dx,dr,d_step_cf):
        
        self.action_len = 5
        self.d_step_cf = d_step_cf

        self.dx = dx
        self.dr = dr
        self.dcf = round(1/d_step_cf,2)
        self.sizes = (3, 3, self.d_step_cf + 1, 4,8)
        self.count = np.prod(self.sizes)
        self.actions = {}
        self.actions_computation()
        self.reset_act_id = 4

    def actions_computation(self):
        idx = 0
        for cnf in range(8):
            for cs in range(4):
                for csf in range(self.d_step_cf + 1):
                    for r in [-self.dr,0,self.dr]:
                        for m in [-self.dx,0,self.dx]:
                            move = m
                            rotate = r
                            chase_focus = float(self.dr*csf)
                            casting_slot = cs
                            change_focus = cnf
                            act=(move,rotate,chase_focus,casting_slot,change_focus)
                            self.actions[idx] = act
                            idx +=1
    
    
    def sample(self, sample=3):
        keys = random.sample(range(0, self.count-1), 3)
        actions = [self.actions[k] for k in keys]
        
        return [keys,actions]
    '''

    def __init__(self,dx,dr):
        
        self.action_len = 5

        self.dx = dx
        self.dr = dr
        self.count = 16
        self.actions = {}
        self.actions_computation()
        self.n_agents = 3
        
    def actions_computation(self):
        self.actions[0] = [0,0,0,0,0]               #do nothing
        self.actions[1] = [self.dx,0,0,0,0]         #move +
        self.actions[2] = [-self.dx,0,0,0,0]        #move -
        self.actions[3] = [0,self.dr,0,0,0]         #rotate +
        self.actions[4] = [0,-self.dr,0,0,0]        #rotate -
        self.actions[5] = [0,0,1,0,0]               #chase focus
        self.actions[6] = [0,0,0,1,0]               #cast ability 1
        self.actions[7] = [0,0,0,2,0]               #cast ability 2
        self.actions[8] = [0,0,0,3,0]               #cast ability 3
        self.actions[9] = [0,0,0,0,1]               # change focus to i = 1
        self.actions[10] = [0,0,0,0,2]              # change focus to i = 2
        self.actions[11] = [0,0,0,0,3]              # change focus to i = 3
        self.actions[12] = [0,0,0,0,4]              # change focus to i = 4
        self.actions[13] = [0,0,0,0,5]              # change focus to i = 5
        self.actions[14] = [0,0,0,0,6]              # change focus to i = 6
        self.actions[15] = [0,0,0,0,7]              # change focus to i = 7
        # with i:   1=focus home statue. 2-3=focus teammates, 4=focus enemy statue, 5-7=focus enemy
    
    def sample(self):
        keys = random.sample(range(0, self.count-1), self.n_agents)
        actions = [self.actions[k] for k in keys]
        
        return [keys,actions]




    

if __name__ == '__main__':
    actionsss = Discrete_actions_space(0.1,0.1,4)
    
    actionsss.sample()





