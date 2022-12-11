from gym_derk.envs import DerkEnv
import numpy as np
import random


#action n is a [num_agent,num_action] matrix,
#each row corresponds to the action taken by that agent and it's a list of dim 5, where:
#[MoveX (-1,1) , Rotate (-1,1), ChaseFocus(0,1), CastingSlot(int in {0..3}), Change_focus(int in {0...7})]


class Discrete_actions_space():#DerkEnv.action_space):
    '''
    def __init__(self,d_step_x,d_step_r,d_step_cf):
        self.d_step_x = d_step_x
        self.d_step_r = d_step_r
        self.d_step_cf = d_step_cf

        self.dx = round(1/d_step_x,2)
        self.dr = round(1/d_step_r,2)
        self.dcf = round(1/d_step_cf,2)

        self.actions = self.actions_computation()
        self.count = len(self.actions)
        self.sizes = (2*self.d_step_x+1, 2*self.d_step_r + 1, 2*self.d_step_cf + 1, 4,8)
    
    def actions_computation(self):
        acts = []
        for a in range(2*self.d_step_x + 1):
            for b in range(2*self.d_step_r + 1):
                for c  in range(self.d_step_r + 1):
                    for e in range(4):
                        for f in range(8):
                            move = -1.0 + self.dx*a 
                            rotate = -1.0 + self.dr*b
                            chase_focus = float(self.dr*c)
                            casting_slot = e
                            change_focus = f
                            acts.append([move,rotate,chase_focus,casting_slot,change_focus])
        return acts

    
    def sample(self):
        move = np.random.choice([-1.0 + self.dx*i for i in range(2*self.d_step_x + 1)])
        rotate = np.random.choice([-1.0 + self.dr*i for i in range(2*self.d_step_r + 1)])
        chase_focus = np.random.choice([0.0 + self.dr*i for i in range(self.d_step_r + 1)])
        casting_slot = np.random.choice(range(4))
        change_focus = np.random.choice(range(8))
        return [move,rotate,chase_focus,casting_slot,change_focus]

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
    def sample(self):
        move = np.random.choice([-self.dx,0,self.dx])
        rotate = np.random.choice([-self.dr,0,self.dr])
        chase_focus = np.random.choice([0.0 + self.dcf*i for i in range(self.d_step_cf + 1)])
        casting_slot = np.random.choice(range(4))
        change_focus = np.random.choice(range(8))
        return [move,rotate,chase_focus,casting_slot,change_focus]
    '''

if __name__ == '__main__':
    actionsss = Discrete_actions_space(0.1,0.1,4)
    
    actionsss.sample()





