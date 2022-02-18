# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 16:05:46 2022

@author: WSY

"""
import numpy as np
import pygame # render the game state
import time # control AI step internal 
import random
import torch
import torch.nn as nn

from agent import DQNAgent


class Game():
    ''' 
    Create a game enviroment for Ferrero game.
    Game area is a 6 plus 8 np.array.
    
    '''
    def __init__(self,
                 ROW=6,
                 COL=8,
                 actions_list=['up', 'down', 'left', 'right'],
                 actions_num=6*8*4,
                 episodes=50000):
        ''' 
        Initialize game.

        Parameters
        ----------
        ROW (int) : Num of rows
        COL (int) : Num of column
        actions_list (str list): Consisting of 4 directions
        actions_num (int) : Num of actions (contain many ilegal actions)
        raw_action (dict): A raw action is represented by position and direction
        raw_action = {'pos': pos (tuple), 'direc': direc (int)}
        state (dict): A state stores observation and legal std_actions
        state = {'obs':obs, 'legal_actions':legal_actions}
        
        Returns
        -------
        None.

        '''
        self.ROW = ROW
        self.COL = COL
        self.actions_list = actions_list
        self.actions_num = actions_num
        self.episodes = episodes
        self.state = {'obs': None, 'legal_actions':None}
        
    def reset(self):
        ''' 
        Reset game state.
        '''
        ROW, COL = self.ROW, self.COL
        self.state['obs'] = np.ones((ROW, COL))
        self.state['obs'][0, 0] = 0
        self.state['legal_actions'] = self.get_legal_actions()
    
        
    def is_end(self):
        '''
        Judge whether a game is end.

        Returns
        -------
        flag (bool)

        '''
        actions = self.get_legal_actions()
        return True if actions == [] else False
    
    
    def get_legal_actions(self):
        '''
        Return all legal std_actions.

        Returns
        -------
        std_actions (list of int)

        '''
        ROW, COL = self.ROW, self.COL
        std_actions = []
        for i in range(ROW):
            for j in range(COL):
                if self.state['obs'][i, j] == 1:
                    if 0 <= i - 2 < ROW:
                        if self.state['obs'][i-1, j] == 1 and self.state['obs'][i-2, j] == 0: # up
                            std_actions.append( (i*COL+j)*4 + 0 )
                    if 0 <= i + 2 < ROW:
                        if self.state['obs'][i+1, j] == 1 and self.state['obs'][i+2, j] == 0: # down
                            std_actions.append( (i*COL+j)*4 + 1 )
                    if 0 <= j - 2 < COL:
                        if self.state['obs'][i, j-1] == 1 and self.state['obs'][i, j-2] == 0: # left
                            std_actions.append( (i*COL+j)*4 + 2 )
                    if 0 <= j + 2 < COL:
                        if self.state['obs'][i, j+1] == 1 and self.state['obs'][i, j+2] == 0: # right 
                            std_actions.append( (i*COL+j)*4 + 3 )
        return std_actions
    
    
    def get_legal_pos(self, pos):
        '''
        Return all legal positions given a selected position

        Parameters
        ----------
        pos (tuple) : (x, y)
            
        Returns
        -------
        legal_pos (list of tuple): [(x1, y1), (x2, y2),...] 

        '''
        ROW, COL = self.ROW, self.COL
        (x, y) = pos
        legal_pos = []
        if self.state['obs'][x, y] == 1:
            if 0 <= x - 2 < ROW:
                if self.state['obs'][x-1, y] == 1 and self.state['obs'][x-2, y] == 0: # up
                    legal_pos.append((x-2, y))
            if 0 <= x + 2 < ROW:
                if self.state['obs'][x+1, y] == 1 and self.state['obs'][x+2, y] == 0: # down
                    legal_pos.append((x+2, y))
            if 0 <= y - 2 < COL:
                if self.state['obs'][x, y-1] == 1 and self.state['obs'][x, y-2] == 0: # left
                    legal_pos.append((x, y-2))
            if 0 <= y + 2 < COL:
                if self.state['obs'][x, y+1] == 1 and self.state['obs'][x, y+2] == 0: # left
                    legal_pos.append((x, y+2))
        return legal_pos
                   
    
    def raw_to_std(self, raw_action):
        '''
        Change a raw action to a standard one.

        Parameters
        ----------
        raw_action (dict): {'pos': pos (tuple), 'direc': direc (int)}

        Returns
        -------
        std_action (int)

        '''
        COL = self.COL
        (x, y), direc = raw_action['pos'], raw_action['direc']
        return ( x*COL + y )*4 + direc
    
    
    def std_to_raw(self, std_action):
        '''
        Change a standard action to a raw one.

        Parameters
        ----------
        std_action (int)

        Returns
        -------
        raw_action (dict): {'pos': pos (tuple), 'direc': direc (int)}

        '''
        COL = self.COL
        direc = std_action % 4
        tmp = std_action // 4
        x, y = tmp // COL, tmp % COL
        return {'pos':(x, y), 'direc':direc}
        
        
    def step(self, std_action):
        '''
        Agent Interact with enviroment based on Morkov model.
        Agent takes an action, recieves reward, and changes envirooment to a new state.

        Parameters
        ----------
        std_action (int) 

        Returns
        -------
        state (dict) : {'obs':obs, 'legal_actions':legal_actions} 
        next_state (dict):  {'obs':obs, 'legal_actions':legal_actions}
        reward (int)
        done (bool)

        '''
        state = self.state
        raw_action = self.std_to_raw(std_action)
        (x, y), direc = raw_action['pos'], raw_action['direc']
        self.state['obs'][x, y] = 0
        if direc == 0: # up
            self.state['obs'][x-1, y] = 0
            self.state['obs'][x-2, y] = 1
        elif direc == 1: # down
            self.state['obs'][x+1, y] = 0
            self.state['obs'][x+2, y] = 1
        elif direc == 2: # left
            self.state['obs'][x, y-1] = 0
            self.state['obs'][x, y-2] = 1
        elif direc == 3: # right
            self.state['obs'][x, y+1] = 0
            self.state['obs'][x, y+2] = 1
        
        next_state = self.state
        next_state['legal_actions'] = self.get_legal_actions()
        
        reward = 0
        done = False
        if self.is_end():
            reward = 8 - self.state['obs'].sum()
            done = True
            self.reset()
        
        return state, next_state, reward, done
        
                
class UserInterface():
    '''
    Use pygame to render game enviroment.
    In general, a game object Game() is created in initialization of the UserInterface object.
    Function run() provide main loop for refreshing the screen,
    which contains 3 basic function: processInput(), update(), and render().
    processInput() : monitor device input in order to store control orders or quit the game.
    update() : execute stored control orders and update game state.
    render() : render game state.
    
    '''
    def __init__(self):
        pygame.init()
        
        # Create Game object
        self.game = Game()
        self.game.reset()
        self.ROW, self.COL = self.game.ROW, self.game.COL
        
        # Create Agent object
        self.agent = DQNAgent()
        
        # General args
        self.BGCOLOR = (106,90,205) # background color
        self.INTERVAL = 0 # AI moving interval
        self.FPS = 40 # frame per second
        self.SIZE = 75 # basic size
        
        # Basic configuration
        self.window = pygame.display.set_mode((self.COL * self.SIZE, self.ROW * self.SIZE)) # Set window size
        pygame.display.set_caption('Ferrero game') # Set window name
        self.running = True # UI running flag
        self.clock = pygame.time.Clock() # FPS control
        self.time = time.time() # record time
        
        # Switch mode
        # AI will move only if human_mode == False and AI_mode == True
        self.human_mode = False # whether allow human action
        self.AI_mode = True # whether allow AI action
        
        # Store human input
        # if select['action'] is not none, it will be executed in update()
        self.select = {'pos':None, 'legal_pos':[], 'action':None}
        
        # Load image
        self.img = pygame.image.load('texture/ferrero1.png').convert_alpha()
        self.img = pygame.transform.smoothscale(self.img, (self.SIZE, self.SIZE))
        self.img_legal = pygame.image.load('texture/legal.png').convert_alpha()
        self.img_legal = pygame.transform.smoothscale(self.img_legal, (0.7*self.SIZE, 0.7*self.SIZE))
        
        # Load font
        self.font = pygame.font.Font('texture/BD_Cartoon_Shout.ttf', 20)
        
        # Create text
        self.select_text = self.font.render('select', True, (220,20,60))
     
        
    def processInput(self):
        for event in pygame.event.get():
            # Press cancel button on the top right corner to quit the game
            if event.type == pygame.QUIT:
                self.running = False
                break
            elif event.type == pygame.MOUSEBUTTONDOWN and self.human_mode: # Press mouse
                x, y = pygame.mouse.get_pos()            
                if event.button == 1: # Press left mouse button to select 
                    print('left click: (%d,%d)' % (x, y))
                    row = y // self.SIZE
                    col = x // self.SIZE
                    
                    if (row, col) in self.select['legal_pos']:
                        x, y = self.select['pos']
                        a = None
                        if row == x - 2 and col == y:
                            a = {'pos':(x,y), 'direc':0}
                        elif row == x + 2 and col == y:
                            a = {'pos':(x,y), 'direc':1}
                        elif row == x and col == y - 2:
                            a = {'pos':(x,y), 'direc':2}
                        elif row == x and col == y + 2:
                            a = {'pos':(x,y), 'direc':3}
                        self.select['action'] = a 
                    
                    else:
                        if self.game.state['obs'][row, col] == 1:
                            self.select['pos'] = (row, col)
                            self.select['legal_pos'] = self.game.get_legal_pos((row, col))
                            print('legal position: ', self.select['legal_pos'])
                    
    
    def update(self):
        if time.time() - self.time < self.INTERVAL:
            return
        
        if self.human_mode == True:
            if self.select['action'] is not None:
                a = self.game.raw_to_std(self.select['action'])
                self.game.step(a)
                self.select = {'pos':None, 'legal_pos':[], 'action':None}
            
        if self.AI_mode == True and self.human_mode == False:
            if self.game.episodes > 0:
                if self.game.is_end() is not True:
                    # state (dict) : {'obs':obs, 'legal_actions':legal_actions} 
                    self.game.state['legal_actions'] = self.game.get_legal_actions()
                    action = self.agent.step(self.game.state)
                    state, next_state, reward, done = self.game.step(action)
                    ts = [state, action, reward, next_state, done]
                    self.agent.feed(ts)
                    if done:
                        self.game.episodes -= 1
                    self.time = time.time()
            else:
                self.running = False
            
    
    def render(self):
        # 填充背景色
        self.window.fill(self.BGCOLOR)
        
        # 显示分隔线
        for i in range(self.ROW + 1):
            LINECOLOR = (72,61,139)
            pygame.draw.line(self.window, LINECOLOR, (0, i*self.SIZE), (self.COL*self.SIZE, i*self.SIZE), 4)
        for i in range(self.COL + 1):
            pygame.draw.line(self.window, LINECOLOR, (i*self.SIZE, 0), (i*self.SIZE, self.ROW*self.SIZE), 4)
        
        # 显示巧克力
        for i in range(self.ROW):
            for j in range(self.COL):
                if self.game.state['obs'][i, j] == 1:
                    self.window.blit(self.img, (j*self.SIZE, i*self.SIZE))
        
        # 显示选中提示
        if self.select['pos'] is not None:
            (x, y) = self.select['pos']
            self.window.blit(self.select_text, (y*self.SIZE , (x+0.4)*self.SIZE))
            # 显示选中格子的合法动作
            for one in self.select['legal_pos']:
                x, y = one
                self.window.blit(self.img_legal, ((y+0.2)*self.SIZE, (x+0.2)*self.SIZE))
            
        
        # pygame刷新显示
        pygame.display.update()
        
    # 主循环
    def run(self):
        while self.running:
            self.processInput()
            self.update()
            self.render()
            self.clock.tick(self.FPS)
        

###########################################################
if __name__ == '__main__':
    UI = UserInterface()
    UI.run()
    pygame.quit()
    







    






















