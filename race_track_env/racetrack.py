import numpy as np
from gymnasium import Env

import os
import pygame

STARTING = 0.8
FINISHING = 0.4

# Race track environment
class RaceTrack(Env):

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, track_map:str,  render_mode:str=None, render_fps:int=10, size:int=2, env_dynamics=False):
        self.size = size
        self.env_name = track_map
        self.metadata.update({'render_fps': render_fps})
        assert track_map in ['a', 'b']
        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode
        # Get the absolute path of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filename = 'track_a.npy' if track_map == 'a' else 'track_b.npy'
        self.env_dynamics = env_dynamics
        
        # Join the current directory with the relative path to the map file
        map_file_path = os.path.join(current_dir, 'maps', filename)
        
        with open(map_file_path, 'rb') as f:
            self.track_map = np.load(f)

        # Initialize parameters for pygame
        self.window_size = self.track_map.shape
        # Pygame's coordinate if the transpose of that of numpy
        self.window_size = (self.window_size[1] * self.size, self.window_size[0] * self.size)
        self.window = None # window for pygame rendering
        self.clock = None # clock for pygame ticks
        self.truncated = False

        # Get start states, choose one to be the root node
        self.start_states = np.dstack(np.where(self.track_map==STARTING))[0]
        self.start_state = self.start_states[np.random.choice(self.start_states.shape[0])]
        self.nS = (*self.track_map.shape, 5, 9) # observation space
        self.nA = 9 # action space
        self.state = None # Initialize state
        self.speed = None # Initialize speed

        # Mapping the integer action to acceleration tuple
        self._action_to_acceleration = {
            0: (-1, -1),
            1: (-1, 0),
            2: (-1, 1),
            3: (0, -1),
            4: (0, 0),
            5: (0, 1),
            6: (1, -1),
            7: (1, 0),
            8: (1, 1)
        }

    # Check if the race car go accross the finishing line
    def _check_finish(self, state):
        finish_states = np.where(self.track_map == FINISHING)
        rows = finish_states[0]
        col = finish_states[1][0]
        if state[0] in rows and state[1] >= col:
            return True
        return False

    # Check if the track run out of the track
    def _check_out_track(self, state):
        row, col = state
        H, W = self.track_map.shape
        # If the car go out of the boundary 
        if row < 0 or row >= H or col < 0 or col >= W:
            return True
        # Check if the car run into the gravels
        if self.track_map[state[0], state[1]] == 0:
            return True
        
        # Check if part of the path run into gravels
        for row_step in range(state[0], row, -1):
            if self.track_map[row_step, state[1]] == 0: return True
        for col_step in range(state[1], col, 1 if col > state[1] else -1):
            if self.track_map[row, col_step] == 0: return True

        return False


    # reset the car to one of the starting positions
    def reset(self):
        # choose the same starting state.
        self.state = self.start_state
        self.speed = (0, 0)
        if self.render_mode == 'human':
            self.render(self.render_mode, self.state)
        return (*self.state, *self.speed)
    
    def get_next_observation(self, observation, action):
        state, speed = observation[:2], observation[2:]
        new_state = np.copy(state)
        y_act, x_act = self._action_to_acceleration[action]
        temp_y_acc = speed[0] + y_act
        temp_x_acc = speed[1] + x_act
        
        if temp_y_acc < -4: temp_y_acc = -4
        if temp_y_acc > 0: temp_y_acc = 0 # Avoid the car from going backward
        if temp_x_acc < -4: temp_x_acc = -4
        if temp_x_acc > 4: temp_x_acc = 4
        
        new_state[0] += temp_y_acc
        new_state[1] += temp_x_acc
        speed = (temp_y_acc, temp_x_acc)
        return (*new_state, *speed)

    # take actions
    def step(self, action):
        # take random action in case environment dynamics is enabled
        if self.env_dynamics:
            if np.random.rand() <= 0.1:
                action = np.random.choice(self.nA)
        observation = self.get_next_observation((*self.state,*self.speed), action)
        new_state = (observation[:2])
        speed = (observation[2:])
        terminated = False   

        # check if next position crosses the finishing line
        if self._check_finish(new_state):
            terminated = True 
        # check if next postion locates in invalid places
        elif self._check_out_track(new_state):
            self.reset()
        else:
            self.state = new_state
            self.speed = speed

        if self.render_mode == 'human':
            self.render(self.render_mode, new_state)
        
        return (*new_state, *speed), -1 , terminated, self.truncated

    # visualize race map
    def render(self, mode, state):

        if self.window is None:
            pygame.init()
            pygame.display.set_caption('Race Track')
            if mode == 'human':
                self.window = pygame.display.set_mode(self.window_size)

        if self.clock is None:
            self.clock = pygame.time.Clock()

        rows, cols = self.track_map.shape
        self.window.fill((255, 255, 255))
        
        # Draw the map
        for row in range(rows):
            for col in range(cols):
                cell_val = self.track_map[row, col]
                # Draw finishing cells
                if cell_val == FINISHING:
                    fill = (235, 52, 52)
                    pygame.draw.rect(self.window, fill, (col * self.size, row * self.size, self.size, self.size), 0)
                # Draw starting cells
                elif cell_val == STARTING:
                    fill = (61, 227, 144)            
                    pygame.draw.rect(self.window, fill, (col * self.size, row * self.size, self.size, self.size), 0)

                color =(120, 120, 120)
                # Draw gravels
                if cell_val == 0:
                    color = (255, 255, 255)
                # Draw race track
                elif cell_val == 1:
                    color = (160, 160, 160)
                
                pygame.draw.rect(self.window, color, (col * self.size, row * self.size, self.size, self.size), 1)
        
        # Draw the car
        pygame.draw.rect(self.window, (86, 61, 227), (state[1] * self.size, state[0] * self.size, self.size, self.size), 0)

        if mode == "human":
            pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.window = None
                    pygame.quit()
                    self.truncated = True
            self.clock.tick(self.metadata['render_fps'])