import numpy as np


class ActionSpace(object):
    def __init__(self, n):
        self.n = n

    def sample(self):
        return np.random.randint(0, self.n)


class ObservationSpace(object):
    def __init__(self, shape):
        self.shape = shape
        self.state_0 = np.random.randint(0, 50, shape, dtype=np.int32)
        self.state_1 = np.random.randint(100, 150, shape, dtype=np.int32)
        self.state_2 = np.random.randint(200, 250, shape, dtype=np.int32)
        self.state_3 = np.random.randint(300, 350, shape, dtype=np.int32)
        self.states = [self.state_0, self.state_1, self.state_2, self.state_3]


class EnvTest(object):
    """
    Adapted from Igor Gitman, CMU / Karan Goel
    Modified
    """

    def __init__(self, shape=(10, 10, 4), high=255):
        # 4 states
        self.rewards = [0.1, -0.3, 0.0, -0.2]
        self.cur_state = 0
        self.num_iters = 0
        self.was_in_second = False
        self.state_shape = lambda: shape
        self.num_actions = lambda: 5
        self._action_space = ActionSpace(5)
        self._observation_space = ObservationSpace(shape)
        self._high = high

    def reset(self):
        self.cur_state = 0
        self.num_iters = 0
        self.was_in_second = False
        return self._observation_space.states[self.cur_state]

    def act(self, action):
        assert 0 <= action <= 4
        self.num_iters += 1
        if action < 4:
            self.cur_state = action
        reward = self.rewards[self.cur_state]
        if self.was_in_second is True:
            reward *= -10
        if self.cur_state == 2:
            self.was_in_second = True
        else:
            self.was_in_second = False
        return (
            # self.observation_space.states[self.cur_state],
            reward,
            self.num_iters >= 5,
            # {"ale.lives": 0},
        )

    def state(self):
        return self._observation_space.states[self.cur_state] / self._high

    def render(self):
        print(self.cur_state)
