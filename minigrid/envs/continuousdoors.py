from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key
from minigrid.minigrid_env import MiniGridEnv
from collections import defaultdict


class ContinuousDoorsEnv(MiniGridEnv):
    """
    ## Description

    This environment has a key that the agent must pick up in order to unlock a
    door and then get to the green goal square. This environment is difficult,
    because of the sparse reward, to solve using classical RL algorithms. It is
    useful to experiment with curiosity or curriculum learning.

    ## Mission Space

    "use the key to open the door and then get to the goal"

    ## Action Space

    | Num | Name         | Action                    |
    |-----|--------------|---------------------------|
    | 0   | left         | Turn left                 |
    | 1   | right        | Turn right                |
    | 2   | forward      | Move forward              |
    | 3   | pickup       | Pick up an object         |
    | 4   | drop         | Unused                    |
    | 5   | toggle       | Toggle/activate an object |
    | 6   | done         | Unused                    |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-ContinuousDoors-8x8-v0`
    - `MiniGrid-ContinuousDoors-12x12-v0`
    - `MiniGrid-ContinuousDoors-16x16-v0`

    """

    def __init__(self, size=8, max_steps: int | None = None, **kwargs):
        if max_steps is None:
            max_steps = 10 * size**2
        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space, grid_size=size, max_steps=max_steps, **kwargs
        )
        self.unlocked_achievement = defaultdict(lambda: False)

    @staticmethod
    def _gen_mission():
        return "use the key to open the continuous doors and then get to the goal"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Create a vertical splitting wall
        # splitIdx = self._rand_int(2, width - 2)
        assert width%2 == 0
        splitIdx = int(2+(width-8)/2)
        for i in range(4):
            self.grid.vert_wall(splitIdx+i, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        doorIdx = self._rand_int(1, height - 2)
        self.put_obj(Door("yellow", is_locked=True), splitIdx, doorIdx)
        self.put_obj(Door("yellow", is_locked=True), splitIdx+1, doorIdx)
        self.put_obj(Door("red", is_locked=True), splitIdx+2, doorIdx)
        self.put_obj(Door("yellow", is_locked=True), splitIdx+3, doorIdx)
        self.first_yellow_door_pos = (splitIdx, doorIdx)
        self.first_red_door_pos = (splitIdx+2, doorIdx)

        # Place a yellow key on the left side
        self.yellow_key_pos = self.place_obj(obj=Key("yellow"), top=(0, 0), size=(splitIdx, height))
        self.red_key_pos = self.place_obj(obj=Key("red"), top=(0, 0), size=(splitIdx, height))

        self.mission = "use the key to open the door and then get to the goal"

    def step(self, action):
        """Steps through the environment with `action`."""
        obs, reward, terminated, truncated, info = super().step(action)

        # env = self.unwrapped
        for pos in [self.yellow_key_pos, self.red_key_pos]:
            if not self.unlocked_achievement[pos]:
                wrld_obj = self.grid.get(*pos)
                if wrld_obj is None:
                    reward = 0.1
                    self.unlocked_achievement[pos] = True

        for pos in [self.first_yellow_door_pos, self.first_red_door_pos]:
            if not self.unlocked_achievement[pos]:
                wrld_obj = self.grid.get(*pos)
                if wrld_obj.is_open:
                    reward = 0.1
                    self.unlocked_achievement[pos] = True

        for idx, pos in enumerate([self.yellow_key_pos, self.first_yellow_door_pos, self.red_key_pos, self.first_red_door_pos]):
            if not self.unlocked_achievement[pos]:
                info["next_skill"] = idx
                break
        if not "next_skill" in info.keys():
            info["next_skill"] = len([self.yellow_key_pos, self.first_yellow_door_pos, self.red_key_pos, self.first_red_door_pos])

        return obs, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.unlocked_achievement = defaultdict(lambda:False)
        return super().reset(seed=seed)