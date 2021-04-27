from mlagents.envs import UnityEnvironment
from enum import Enum
#from skimage.transform import resize
import skimage.transform
import numpy as np


class Constant:
    STATE_HEIGHT = 14
    STATE_WIDTH = 17
    STATE_WIDTH_BOUND = STATE_WIDTH - 1
    SCALE_X = 25
    SCALE_Y = 30


class StateKey(Enum):
    EMPTY = 1
    FROG = 2
    RED_CAR = 3
    LOG = 4
    PLATFORM = 5
    WIN = 6
    HAZARD_ZONE = 7
    START = 8
    WALL = 9
    TUNNEL_ON_KEY = 10
    BLUE_CAR = 11
	
def create_sym_rep(observation_data):
    # frog in goal and witch ine is it ?
    frog_in_goal = int(observation_data[33])  # 0- if none , 1 - for right most one and 2 for left one
    # is car in tunnel
    car_in_tunnel = int(observation_data[32])  # 1 if car is inside the tunnel
    # frog position send them as the index for the state array
    # (12,8) will be the start position
    # to calculate x position (original x value - 50) / 25 and round this to nearest int (coz of the logs)
    # y should be fixed and when it go up or down +1 or - 1
    frog_x = int(observation_data[0])
    frog_y = int(observation_data[1])

    current_state = [[0] * Constant.STATE_WIDTH for i in range(Constant.STATE_HEIGHT)]

    # reset state part

    # init all to empty
    # only need to do this from 5 - 11 index (road part)

    for i in range(5, 12):
        for j in range(0, Constant.STATE_WIDTH):
            current_state[i][j] = StateKey.EMPTY.value

    # add platform and start pos
    # 13 - start , 12,4 platform
    for j in range(0, Constant.STATE_WIDTH):
        current_state[13][j] = StateKey.START.value
        current_state[12][j] = StateKey.PLATFORM.value
        current_state[4][j] = StateKey.PLATFORM.value

    # fill with hazard zone
    # rows 0 - 3 and col - all
    for i in range(0, 4):
        for j in range(0, Constant.STATE_WIDTH):
            current_state[i][j] = StateKey.HAZARD_ZONE.value

    # add goal positions
    # if frog is in the goal mark it as hazard ???
    # goal pos 1 is in 5 - 6 col in 0 th row
    for j in range(5, 7):
        # frog is in 1st goal pos
        if frog_in_goal == 1:
            current_state[0][j] = StateKey.HAZARD_ZONE.value
        else:
            current_state[0][j] = StateKey.WIN.value

    # goal pos 2 is in 10 - 11 col in 0 th row
    for j in range(10, 12):
        # frog is in 2nd goal pos
        if frog_in_goal == 2:
            current_state[0][j] = StateKey.HAZARD_ZONE.value
        else:
            current_state[0][j] = StateKey.WIN.value

    # mark tunnel
    # if car in tunnel mark it as lit  ow as a wall
    # its in row 8 col [6-12]
    for j in range(6, 13):
        # frog is in 2nd goal pos
        if car_in_tunnel == 1:
            current_state[8][j] = StateKey.TUNNEL_ON_KEY.value
        else:
            current_state[8][j] = StateKey.WALL.value

    # frog position
    # from game I can directly send frog x , y as a index for this array
    current_state[frog_y][frog_x] = StateKey.FROG.value

    # Add cars / logs
    # array  observations [2 - 20]
    # s , x , z

    index = 1  # easy to travel the array 1st car y data at 4 th index
    for i in range(0, 10):
        index += 3
        x_pos = int(observation_data[index - 1])
        y_pos = int(observation_data[index])
        size = int(observation_data[index - 2])

        # out of view car /log
        if x_pos + size <= 0 or x_pos > 16:
            continue
        elif x_pos < 0 < x_pos + size:
            size = size + x_pos
            x_pos = 0
        # car in tunnel lane
        if y_pos == 8:
            # car is fully in tunnel
            if x_pos >= 6 and x_pos + size <= 12:
                continue
            # car is left and in tunnel
            elif x_pos < 6 <= x_pos + size:
                for j in range(x_pos, 6):
                    current_state[y_pos][j] = StateKey.RED_CAR.value
            elif x_pos <= 12 < x_pos + size:
                for j in range(13, x_pos + size):
                    current_state[y_pos][j] = StateKey.RED_CAR.value
            # out of tunnel
            else:
                for j in range(x_pos, x_pos + size):
                    if j > 16:
                        break
                    current_state[y_pos][j] = StateKey.RED_CAR.value

        # all other cars /logs
        elif index <= 32:
            # todo : get car color from game
            # for blue cars
            if y_pos == 5 or y_pos == 7 or y_pos == 9 or y_pos == 10:
                for j in range(x_pos, x_pos + size):
                    if j > 16:
                        break
                    current_state[y_pos][j] = StateKey.BLUE_CAR.value
            # for red cars
            elif y_pos == 6 or y_pos == 8 or y_pos == 11:
                for j in range(x_pos, x_pos + size):
                    if j > 16:
                        break
                    current_state[y_pos][j] = StateKey.RED_CAR.value
            # logs
            else:
                for j in range(x_pos, x_pos + size):
                    if j > 16:
                        break
                    current_state[y_pos][j] = StateKey.LOG.value

    return  current_state
	
class Game:

    """
    set up unity ml agent environment
    @:param game_location  : file path for executable
    """
    def __init__(self, game_location):
        self.ENV_LOCATION = game_location
        self.load_env(0)

    """
    load unity environment
    @:param wid  : id for the worker in unity environment 
    """
    def load_env(self, wid):
        # load env
        env_name = self.ENV_LOCATION
        self.env = UnityEnvironment(env_name, worker_id=wid)
        # Set the default brain to work with
        self.default_brain = self.env.brain_names[0]
        self.brain = self.env.brains[self.default_brain]
        # Reset the environment - train mode enabled
        env_info = self.env.reset(train_mode=True)[self.default_brain]

    # this frogger game action space is 5, actions[0] = selected action (action = [[1]])
    # actions
    # 1 - up, 2 - down , 3- left , 4 -right , 0 - do nothing
    """
    performs a given action to the unity game 
    @:param action_value : action to be execute
    @:param image_height : Desire image height 
    @:param image_width  : Desire image width 
    @:param number_of_frames : stack size (this number of frames will e stack together by performing no op action )
    @:return reward : reward for the action 
    @:return stack  : stack of frames
    @:return terminal : if game reached terminal state or not
    @:return observation_data : return data that need for sym reps
    """
    def perform_action(self, action_value, image_height, image_width, number_of_frames=4):
        #print("action_value: ",action_value)
        action = [[0]]
        #print("action: ",action)
        action[0] = action_value
        #print("action[0]: ",action[0])
        terminal = False  # indication of terminal state
        # 3 - R, G, B
        #size = (image_height, image_width, 3, number_of_frames)  # create list to keep frames
        size = (number_of_frames,3,image_height, image_width)
        stack = np.zeros(size)
        # to store data to sym rep 32 data is send from game
        observation_data = [[0] * 34 for i in range(number_of_frames)]

        # first frame after action
        env_info = self.env.step(action)[self.default_brain]  # send action to brain
        reward = round(env_info.rewards[0], 5)  # get reward
        new_state = env_info.visual_observations[0][0]  # get state visual observation
		#print("new_state keras shape: ", new_state_gray.shape)
        #print("new_state keras shape: ", new_state.shape)
        observations = env_info.vector_observations  # get vector observations
        observation_data[0] = observations[0]
        # new_state_gray = skimage.color.rgb2gray(new_state)  # covert to gray scale
        new_state_gray = skimage.transform.resize(new_state, (image_height, image_width))  # resize
        #print("new_state_gray shape: ",new_state_gray.shape)
        new_state_reshape = np.reshape(new_state_gray, (3,100,100))
        #print("new_state keras shape: ", new_state_reshape.shape)
        # check terminal reached
        if env_info.local_done[0]:
            terminal = True

        # add the state to the 0 th position of stack
        stack[0, :, :, :] = new_state_reshape

        # get stack of frames after the action
        for i in range(1, number_of_frames):
            env_info = self.env.step()[self.default_brain]  # change environment to next step without action
            st = env_info.visual_observations[0][0]
			
            observations = env_info.vector_observations  # get vector observations
            observation_data[i] = observations[0]
            #st_gray = skimage.color.rgb2gray(st)
            st_gray = skimage.transform.resize(st, (image_height, image_width))
            st_gray_reshape = np.reshape(new_state_gray, (3,100,100))
            stack[i,:, :, :] = st_gray_reshape
            # if terminal only consider the reward for terminal
            if env_info.local_done[0]:
                terminal = True
                reward = round(env_info.rewards[0], 5)

        # reshape for Keras
        # noinspection PyArgumentList
        #stack = stack.reshape(1, stack.shape[0], stack.shape[1], stack.shape[2], stack.shape[3])

        #return reward, stack, terminal, observation_data
        return reward, stack, terminal
		

    """
    close environment
    """
    def close(self):
        self.env.close()

    """
    Reset environment 
    """
    def reset(self):
        self.close()
        self.load_env(0)
		
#game_location  = "windows_build/UnityFrogger"
#new_game = Game(game_location)
#action = 0# action to be send to the game 

# image size that you need to get as out put 
#image_height  = 500 
#image_width   = 500 
#reward,stack,terminal,observation_data = new_game.perform_action(action, image_height, image_width)