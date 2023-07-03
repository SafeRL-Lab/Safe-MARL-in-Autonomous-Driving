from typing import Dict, Text, Tuple

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.objects import Obstacle


class MergeEnv(AbstractEnv):

    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "collision_reward": -1,
            "right_lane_reward": 0.1,
            "high_speed_reward": 0.2,
            "merging_speed_reward": -0.5,
            "lane_change_reward": -0.05,
            "on_road_reward": 1,
        })
        return cfg

    def _reward(self, action: int) -> float:
        """
        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions

        But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.

        :param action: the action performed
        :return: the reward of the state-action transition
        """
        reward = sum(self.config.get(name, 0) * reward for name, reward in self._rewards(action).items())
        return utils.lmap(reward,
                          [self.config["collision_reward"] + self.config["merging_speed_reward"],
                           self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                          [0, 1])

    def _rewards(self, action: int) -> Dict[Text, float]:
        return {
            "collision_reward": self.vehicle.crashed,
            "right_lane_reward": self.vehicle.lane_index[2] / 1,
            "high_speed_reward": self.vehicle.speed_index / (self.vehicle.target_speeds.size - 1),
            "lane_change_reward": action in [0, 2],
            "merging_speed_reward": sum(  # Altruistic penalty
                (vehicle.target_speed - vehicle.speed) / vehicle.target_speed
                for vehicle in self.road.vehicles
                if vehicle.lane_index == ("b", "c", 2) and isinstance(vehicle, ControlledVehicle)
            )
        }

    # def _is_terminated(self) -> bool:
    #     """The episode is over when a collision occurs or when the access ramp has been passed."""
    #     return self.vehicle.crashed or bool(self.vehicle.position[0] > 370)
    
    def leader_is_terminal(self, vehicle):
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return vehicle.crashed or bool(vehicle.position[0] > 350) or self.time >= self.config["duration"]
    
    def follower_is_terminal(self, vehicle):
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return vehicle.crashed or bool(vehicle.position[0] > 350) or self.time >= self.config["duration"]
    
    def leader_agend_reward(self, vehicle, action):
        reward = 0
        if vehicle.speed_index / (vehicle.target_speeds.size - 1):
            reward += 2
        
        if vehicle.crashed:
            reward -= 5

        if vehicle.position[0] > 350:
            if self.leader_arrived == False and self.follower_arrived == False:
                reward += 10
                self.first_arrived = 1
            elif self.leader_arrived == False and self.follower_arrived == True:
                reward += 5
            self.leader_arrived = True
        return reward
    
    def follower_agend_reward(self, vehicle, action):
        reward = 0
        if vehicle.speed_index / (vehicle.target_speeds.size - 1):
            reward += 2
        
        if vehicle.crashed:
            reward -= 5
            
        if vehicle.position[0] > 350:
            if self.leader_arrived == False and self.follower_arrived == False:
                reward += 10
                self.first_arrived = 2
            elif self.leader_arrived == True and self.follower_arrived == False:
                reward += 5
            self.follower_arrived = True
        return reward

    def _is_truncated(self) -> bool:
        return False

    def _reset(self) -> None:
        self.leader_arrived = False
        self.follower_arrived = False
        self.first_arrived = 0
        self._make_road()
        self._make_vehicles()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        # simulation
        self.time += 1 / self.config["policy_frequency"]
        self._simulate(action)
        
        # observation
        obs = self.observation_type.observe()
        
        truncated = self._is_truncated()

        # terminate
        leader_terminated = self.leader_is_terminal(self.controlled_vehicles[0])
        follower_terminated = self.follower_is_terminal(self.controlled_vehicles[1])
        terminated = [leader_terminated, follower_terminated]

        # reward
        leader_reward = self.leader_agend_reward(self.controlled_vehicles[0], action[0])
        follower_reward = self.follower_agend_reward(self.controlled_vehicles[1], action[1])   
        reward = [leader_reward, follower_reward]       

        # info
        info = {}
        info["crash"] = self.controlled_vehicles[0].crashed or self.controlled_vehicles[1].crashed
        if info["crash"]:
                info["leader_arrived"] = 0
                info["follower_arrived"] = 0
        else:
            info["leader_arrived"] = self.leader_arrived
            info["follower_arrived"] = self.follower_arrived
        # info = self._info(obs, action)
        # info["leader_arrived"] = self.has_arrived_target(self.controlled_vehicles[0])
        # info["follower_arrived"] = self.has_arrived_target(self.controlled_vehicles[1])

        # cost
        cost = np.zeros(2)
        cost[0] += 5*self.controlled_vehicles[0].crashed
        cost[1] += 5*self.controlled_vehicles[1].crashed
        info["cost"] = cost

        return obs, reward, terminated, truncated, info

    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.

        :return: the road
        """
        net = RoadNetwork()

        # Highway lanes
        ends = [150, 80, 80, 150]  # Before, converging, merge, after
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH]
        line_type = [[c, s], [n, c]]
        line_type_merge = [[c, s], [n, s]]
        for i in range(2):
            net.add_lane("a", "b", StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i]))
            net.add_lane("b", "c", StraightLane([sum(ends[:2]), y[i]], [sum(ends[:3]), y[i]], line_types=line_type_merge[i]))
            net.add_lane("c", "d", StraightLane([sum(ends[:3]), y[i]], [sum(ends), y[i]], line_types=line_type[i]))

        # Merging lane
        amplitude = 3.25
        ljk = StraightLane([0, 6.5 + 4 + 4], [ends[0], 6.5 + 4 + 4], line_types=[c, c], forbidden=True)
        lkb = SineLane(ljk.position(ends[0], -amplitude), ljk.position(sum(ends[:2]), -amplitude),
                       amplitude, 2 * np.pi / (2*ends[1]), np.pi / 2, line_types=[c, c], forbidden=True)
        lbc = StraightLane(lkb.position(ends[1], 0), lkb.position(ends[1], 0) + [ends[2], 0],
                           line_types=[n, c], forbidden=True)
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        self.controlled_vehicles = []
        road = self.road
        ego_vehicle = self.action_type.vehicle_class(road, road.network.get_lane(("a", "b", 1)).position(70, 0), speed=25)
        road.vehicles.append(ego_vehicle)
        self.controlled_vehicles.append(ego_vehicle)
        # other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        # road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(90, 0), speed=29))
        # road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 1)).position(70, 0), speed=31))
        # road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(5, 0), speed=31.5))

        merging_v = self.action_type.vehicle_class(road, road.network.get_lane(("j", "k", 0)).position(70, 0), speed=25)
        # merging_v.target_speed = 30
        road.vehicles.append(merging_v)
        self.controlled_vehicles.append(merging_v)
        # self.vehicle = ego_vehicle
