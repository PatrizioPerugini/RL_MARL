from gym_derk.envs import DerkEnv
import random
import numpy as np


class CustomEnvironment():
    def __init__(self,dx=0.5,dr=0.2,training_mode =False):
        self.ht_primary_col = '#FF0000' #red    
        self.ht_secondary_col = '#FFA500' #orange
        self.at_primary_col = '#ADD8E6' #light blue
        self.at_secondary_col = '#FFFFFF' #white 
        self.arms = ['Talons','BloodClaws','Cleavers',
                    'Cripplers','Pistol','Magnum','Blaster']
        self.miscs = ['FrogLegs','IronBubblegum',
                        'HeliumBubblegum','Shell','Trombone']
        self.tails = ['VampireGland','ParalyzingDart']
        self.healing_tail = ['HealingGland']

        if training_mode:
            self.home_team_conf =  self.team_conf_4_training('home')
            self.away_team_conf = self.team_conf_4_training('away')
        else:
            self.home_team_conf = self.team_conf_4_evaluation('home')
            self.away_team_conf = self.team_conf_4_evaluation('away')

        self.env = DerkEnv( turbo_mode=True,
                home_team = self.home_team_conf,
                away_team = self.away_team_conf)
        
        self.action_space = Discrete_actions_space(dx,dr)
        


    def team_conf_4_evaluation(self,team):
        player_1_slots = [
            random.sample(self.arms,1)[0],
            random.sample(self.miscs,1)[0],
            random.sample(self.tails,1)[0]
        ]
        player_2_slots = [
            random.sample(self.arms,1)[0],
            random.sample(self.miscs,1)[0],
            random.sample(self.tails,1)[0]
        ]
        healer_player_slots = [
            random.sample(self.arms,1)[0],
            random.sample(self.miscs,1)[0],
            self.healing_tail
        ]
        if team == 'home':
            primary_color = self.ht_primary_col
            secondary_color = self.ht_secondary_col
        else:
            primary_color = self.at_primary_col
            secondary_color = self.at_secondary_col

        team_conf = [
            {'primaryColor': primary_color,
             'secondaryColor':secondary_color,
             'slots':player_1_slots,
             'rewardFunction':  { 'damageEnemyUnit': 0.1 ,
                                 'damageEnemyStatue': 0.2 ,
                                 'friendlyFire': -0.1,
                                 'killEnemyStatue': 4,
                                 'killEnemyUnit': 1}
            },
            {'primaryColor': primary_color,
             'secondaryColor':secondary_color,
             'slots':healer_player_slots,
             'rewardFunction':  { 'healTeammate1': 0.1 ,
                                 'healTeammate2': 0.1 ,
                                 'healFriendlyStatue': 0.2 ,
                                 'healEnemy': -0.1 ,
                                 'friendlyFire': -0.1,
                                 'killEnemyStatue': 4,
                                 'killEnemyUnit': 1}
            },
            {'primaryColor': primary_color,
             'secondaryColor':secondary_color,
             'slots':player_2_slots,
             'backSpikes':3,
             'rewardFunction':  { 'damageEnemyUnit': 0.1 ,
                                 'damageEnemyStatue': 0.2 ,
                                 'friendlyFire': -0.1,
                                 'killEnemyStatue': 4,
                                 'killEnemyUnit': 1}
            }]
            
        return team_conf
    
    def team_conf_4_training(self,team):
        if team == 'home':
            primary_color = self.ht_primary_col
            secondary_color = self.ht_secondary_col
        else:
            primary_color = self.at_primary_col
            secondary_color = self.at_secondary_col
        generic_player = {'primaryColor': primary_color,
                        'secondaryColor':secondary_color,
                        'rewardFunction': 
                        {'damageEnemyUnit': 1 ,
                        'damageEnemyStatue': 2 ,
                        'healTeammate1': 1 ,
                        'healTeammate2': 1 ,
                        'healFriendlyStatue': 2 ,
                        'healEnemy': -5 ,
                        'friendlyFire': -5,
                        'killEnemyStatue': 40,
                        'killEnemyUnit': 10}}
        
        team_conf = [generic_player,generic_player,generic_player]

        return team_conf

    def reset(self):
        self.env.reset()


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



