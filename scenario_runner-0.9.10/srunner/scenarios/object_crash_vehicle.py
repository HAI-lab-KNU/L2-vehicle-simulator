#!/usr/bin/env python
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Object crash without prior vehicle action scenario:
The scenario realizes the user controlled ego vehicle
moving along the road and encountering a cyclist ahead.
"""

from __future__ import print_function

import math
import random
import time

import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (AtomicBehavior,
                                                                      ActorTransformSetter,
                                                                      ActorDestroy,
                                                                      AccelerateToVelocity,
                                                                      HandBrakeVehicle,
                                                                      KeepVelocity,
                                                                      StopVehicle)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToLocationAlongRoute,
                                                                               InTimeToArrivalToVehicle,
                                                                               DriveDistance)
from srunner.scenariomanager.timer import TimeOut
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import get_location_in_distance_from_wp

import logging

class LoggingProgress(AtomicBehavior):

    def __init__(self, msg, name="Logging"):
        super(LoggingProgress, self).__init__(name)
        self.msg = msg

    def update(self):
        logging.info(self.msg)
        return py_trees.common.Status.SUCCESS

class StationaryObjectCrossing(BasicScenario):

    """
    This class holds everything required for a simple object crash
    without prior vehicle action involving a vehicle and a cyclist.
    The ego vehicle is passing through a road and encounters
    a stationary cyclist.

    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=60):
        """
        Setup all relevant parameters and create scenario
        """
        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        # ego vehicle parameters
        self._ego_vehicle_distance_driven = 40

        # other vehicle parameters
        self._other_actor_target_velocity = 10
        # Timeout of scenario in seconds
        self.timeout = timeout

        super(StationaryObjectCrossing, self).__init__("Stationaryobjectcrossing",
                                                       ego_vehicles,
                                                       config,
                                                       world,
                                                       debug_mode,
                                                       criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        _start_distance = 40
        lane_width = self._reference_waypoint.lane_width
        location, _ = get_location_in_distance_from_wp(self._reference_waypoint, _start_distance)
        waypoint = self._wmap.get_waypoint(location)
        offset = {"orientation": 270, "position": 90, "z": 0.4, "k": 0.2}
        position_yaw = waypoint.transform.rotation.yaw + offset['position']
        orientation_yaw = waypoint.transform.rotation.yaw + offset['orientation']
        offset_location = carla.Location(
            offset['k'] * lane_width * math.cos(math.radians(position_yaw)),
            offset['k'] * lane_width * math.sin(math.radians(position_yaw)))
        location += offset_location
        location.z += offset['z']
        self.transform = carla.Transform(location, carla.Rotation(yaw=orientation_yaw))
        static = CarlaDataProvider.request_new_actor('static.prop.container', self.transform)
        static.set_simulate_physics(True)
        self.other_actors.append(static)

    def _create_behavior(self):
        """
        Only behavior here is to wait
        """
        lane_width = self.ego_vehicles[0].get_world().get_map().get_waypoint(
            self.ego_vehicles[0].get_location()).lane_width
        lane_width = lane_width + (1.25 * lane_width)

        # leaf nodes
        actor_stand = TimeOut(15)
        actor_removed = ActorDestroy(self.other_actors[0])
        end_condition = DriveDistance(self.ego_vehicles[0], self._ego_vehicle_distance_driven)

        # non leaf nodes
        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        scenario_sequence = py_trees.composites.Sequence()

        # building tree
        root.add_child(scenario_sequence)
        scenario_sequence.add_child(ActorTransformSetter(self.other_actors[0], self.transform))
        scenario_sequence.add_child(actor_stand)
        scenario_sequence.add_child(actor_removed)
        scenario_sequence.add_child(end_condition)

        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        collision_criterion = CollisionTest(self.ego_vehicles[0])
        criteria.append(collision_criterion)

        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()


class DynamicObjectCrossing(BasicScenario):

    """
    This class holds everything required for a simple object crash
    without prior vehicle action involving a vehicle and a cyclist/pedestrian,
    The ego vehicle is passing through a road,
    And encounters a cyclist/pedestrian crossing the road.

    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False,
                 debug_mode=False, criteria_enable=True, adversary_type=False, timeout=60):
        """
        Setup all relevant parameters and create scenario
        """
        self._wmap = CarlaDataProvider.get_map()

        # jiwoo 트리거 포인트
        #config.trigger_points[0].location.x = 39.003081
        #config.trigger_points[0].location.y = -166.008240

        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)

        # ego vehicle parameters
        self._ego_vehicle_distance_driven = 40
        # other vehicle parameters
        self._other_actor_target_velocity = 5
        self._other_actor_max_brake = 1.0
        self._time_to_reach = 10
        self._adversary_type = adversary_type  # flag to select either pedestrian (False) or cyclist (True)
        self._walker_yaw = 0
        self._num_lane_changes = 1
        self.transform = None
        self.timeout = timeout
        self._trigger_location = config.trigger_points[0].location
        # Total Number of attempts to relocate a vehicle before spawning
        self._number_of_attempts = 20
        # Number of attempts made so far
        self._spawn_attempted = 0

        self._ego_route = CarlaDataProvider.get_ego_vehicle_route()

        super(DynamicObjectCrossing, self).__init__("DynamicObjectCrossing",
                                                    ego_vehicles,
                                                    config,
                                                    world,
                                                    debug_mode,
                                                    criteria_enable=criteria_enable)

    def _calculate_base_transform(self, _start_distance, waypoint):

        lane_width = waypoint.lane_width

        # Patches false junctions
        if self._reference_waypoint.is_junction:
            stop_at_junction = False
        else:
            stop_at_junction = True

        location, _ = get_location_in_distance_from_wp(waypoint, _start_distance, stop_at_junction)
        waypoint = self._wmap.get_waypoint(location)
        offset = {"orientation": 270, "position": 90, "z": 0.6, "k": 1.0}
        position_yaw = waypoint.transform.rotation.yaw + offset['position']
        orientation_yaw = waypoint.transform.rotation.yaw + offset['orientation']
        offset_location = carla.Location(
            offset['k'] * lane_width * 1.1 * math.cos(math.radians(position_yaw)),
            offset['k'] * lane_width * 1.1 * math.sin(math.radians(position_yaw)))
        location += offset_location
        location.z = self._trigger_location.z + offset['z']
        return carla.Transform(location, carla.Rotation(yaw=orientation_yaw)), orientation_yaw

    def _spawn_adversary(self, transform, orientation_yaw):

        self._time_to_reach *= self._num_lane_changes
        if self._adversary_type is False:
            self._walker_yaw = orientation_yaw
            self._other_actor_target_velocity = 3 + (0.4 * self._num_lane_changes) + 10 # jiwoo 언제 뛰쳐나가는지
            walker_ids = ['walker.pedestrian.0001', 'walker.pedestrian.0002', 'walker.pedestrian.0003', 'walker.pedestrian.0004', 'walker.pedestrian.0005', 'walker.pedestrian.0006', 'walker.pedestrian.0007', 'walker.pedestrian.0008', 'walker.pedestrian.0015', 'walker.pedestrian.0016', 'walker.pedestrian.0017', 'walker.pedestrian.0018', 'walker.pedestrian.0019', 'walker.pedestrian.0020']
            walker = CarlaDataProvider.request_new_actor(random.choice(walker_ids), transform)
            adversary = walker
        else:
            self._other_actor_target_velocity = self._other_actor_target_velocity * self._num_lane_changes
            first_vehicle = CarlaDataProvider.request_new_actor('vehicle.diamondback.century', transform)
            first_vehicle.set_simulate_physics(enabled=False)
            adversary = first_vehicle
        return adversary

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        # cyclist transform
        _start_distance = 12   #jiwoo self.transform이 생성되는 위치와 egovehicle 사이의 거리
        # We start by getting and waypoint in the closest sidewalk.
        waypoint = self._reference_waypoint
        while True:
            wp_next = waypoint.get_right_lane()
            self._num_lane_changes += 1
            if wp_next is None or wp_next.lane_type == carla.LaneType.Sidewalk:
                break
            elif wp_next.lane_type == carla.LaneType.Shoulder:
                # Filter Parkings considered as Shoulders
                if wp_next.lane_width > 2:
                    _start_distance += 1.5
                    waypoint = wp_next
                break
            else:
                _start_distance += 1.5
                waypoint = wp_next

        while True:  # We keep trying to spawn avoiding props

            try:
                self.transform, orientation_yaw = self._calculate_base_transform(_start_distance, waypoint)
                logging.info("transform of walker is "+str(self.transform))
                first_vehicle = self._spawn_adversary(self.transform, orientation_yaw)
                break
            except RuntimeError as r:
                # We keep retrying until we spawn
                print("Base transform is blocking objects ", self.transform)
                _start_distance += 0.4
                self._spawn_attempted += 1
                if self._spawn_attempted >= self._number_of_attempts:
                    raise r

        # Now that we found a possible position we just put the vehicle to the underground
        disp_transform = carla.Transform(
            carla.Location(self.transform.location.x,
                           self.transform.location.y,
                           self.transform.location.z - 500),
            self.transform.rotation)


        first_vehicle.set_transform(disp_transform)
        first_vehicle.set_simulate_physics(enabled=False)
        self.other_actors.append(first_vehicle)

    def _create_behavior(self):
        """
        After invoking this scenario, cyclist will wait for the user
        controlled vehicle to enter trigger distance region,
        the cyclist starts crossing the road once the condition meets,
        then after 60 seconds, a timeout stops the scenario
        """

        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="OccludedObjectCrossing")
        lane_width = self._reference_waypoint.lane_width
        lane_width = lane_width + (1.10 * lane_width * self._num_lane_changes)

        dist_to_trigger = 28 + self._num_lane_changes #+ lane_width #
        # leaf nodes
        if self._ego_route is not None:
            start_condition = InTriggerDistanceToLocationAlongRoute(self.ego_vehicles[0],
                                                                    self._ego_route,
                                                                    self.transform.location,
                                                                    dist_to_trigger)
        else:
            start_condition = InTimeToArrivalToVehicle(self.ego_vehicles[0],
                                                       self.other_actors[0],
                                                       self._time_to_reach)

        actor_velocity = KeepVelocity(self.other_actors[0],
                                      self._other_actor_target_velocity,
                                      name="walker velocity")
        actor_drive = DriveDistance(self.other_actors[0],
                                    0.25 * lane_width,
                                    name="walker drive distance")
        actor_start_cross_lane = AccelerateToVelocity(self.other_actors[0],
                                                      1.0,
                                                      self._other_actor_target_velocity,
                                                      name="walker crossing lane accelerate velocity")
        actor_cross_lane = DriveDistance(self.other_actors[0],
                                         0.8 * lane_width,
                                         name="walker drive distance for lane crossing ")
        actor_stop_crossed_lane = StopVehicle(self.other_actors[0],
                                              self._other_actor_max_brake,
                                              name="walker stop")
        ego_pass_machine = DriveDistance(self.ego_vehicles[0],
                                         5,
                                         name="ego vehicle passed prop")
        actor_remove = ActorDestroy(self.other_actors[0],
                                    name="Destroying walker")
        end_condition = DriveDistance(self.ego_vehicles[0],
                                      self._ego_vehicle_distance_driven,
                                      name="End condition ego drive distance")

        # non leaf nodes

        scenario_sequence = py_trees.composites.Sequence()
        keep_velocity_other = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="keep velocity other")
        keep_velocity = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="keep velocity")

        # building tree

        root.add_child(scenario_sequence)
        scenario_sequence.add_child(LoggingProgress("Right Walker Crossing Scenario"))
        scenario_sequence.add_child(ActorTransformSetter(self.other_actors[0], self.transform,
                                                         name='TransformSetterTS3walker'))
        scenario_sequence.add_child(HandBrakeVehicle(self.other_actors[0], True))
        scenario_sequence.add_child(start_condition)
        scenario_sequence.add_child(HandBrakeVehicle(self.other_actors[0], False))
        scenario_sequence.add_child(LoggingProgress("walker crossed"))
        scenario_sequence.add_child(keep_velocity)
        #scenario_sequence.add_child(keep_velocity_other) jiwoo 이거랑
        scenario_sequence.add_child(actor_stop_crossed_lane)
        scenario_sequence.add_child(LoggingProgress("walker stopped"))
        #scenario_sequence.add_child(actor_remove)  이걸 없애서 가만히 멈춰서있게 구현
        #scenario_sequence.add_child(static_remove)
        scenario_sequence.add_child(end_condition)

        keep_velocity.add_child(actor_velocity)
        keep_velocity.add_child(actor_drive)
        keep_velocity_other.add_child(actor_start_cross_lane)
        keep_velocity_other.add_child(actor_cross_lane)
        keep_velocity_other.add_child(ego_pass_machine)

        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        collision_criterion = CollisionTest(self.ego_vehicles[0])
        criteria.append(collision_criterion)

        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()

#jiwoo
class DynamicObjectCrossingLeft(BasicScenario):

    """
    This class holds everything required for a simple object crash
    without prior vehicle action involving a vehicle and a cyclist/pedestrian,
    The ego vehicle is passing through a road,
    And encounters a cyclist/pedestrian crossing the road.

    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False,
                 debug_mode=False, criteria_enable=True, adversary_type=False, timeout=60):
        """
        Setup all relevant parameters and create scenario
        """
        self._wmap = CarlaDataProvider.get_map()

        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)

        # ego vehicle parameters
        self._ego_vehicle_distance_driven = 40
        # other vehicle parameters
        self._other_actor_target_velocity = 5
        self._other_actor_max_brake = 1.0
        self._time_to_reach = 10
        self._adversary_type = adversary_type  # flag to select either pedestrian (False) or cyclist (True)
        self._walker_yaw = 0
        self._num_lane_changes = 1
        self.transform = None
        self.timeout = timeout
        self._trigger_location = config.trigger_points[0].location
        # Total Number of attempts to relocate a vehicle before spawning
        self._number_of_attempts = 20
        # Number of attempts made so far
        self._spawn_attempted = 0

        self._ego_route = CarlaDataProvider.get_ego_vehicle_route()

        super(DynamicObjectCrossingLeft, self).__init__("DynamicObjectCrossingLeft",
                                                    ego_vehicles,
                                                    config,
                                                    world,
                                                    debug_mode,
                                                    criteria_enable=criteria_enable)

    def _calculate_base_transform(self, _start_distance, waypoint):

        lane_width = waypoint.lane_width

        # Patches false junctions
        if self._reference_waypoint.is_junction:
            stop_at_junction = False
        else:
            stop_at_junction = True

        location, _ = get_location_in_distance_from_wp(waypoint, _start_distance, stop_at_junction)
        waypoint = self._wmap.get_waypoint(location)
        offset = {"orientation": -270, "position": -90, "z": 0.6, "k": 1.0}
        position_yaw = waypoint.transform.rotation.yaw + offset['position']
        orientation_yaw = waypoint.transform.rotation.yaw + offset['orientation']
        offset_location = carla.Location(
            offset['k'] * lane_width * 2 * math.cos(math.radians(position_yaw)),
            offset['k'] * lane_width * 2 * math.sin(math.radians(position_yaw)))
        location += offset_location
        location.z = self._trigger_location.z + offset['z']
        return carla.Transform(location, carla.Rotation(yaw=orientation_yaw)), orientation_yaw

    def _spawn_adversary(self, transform, orientation_yaw):

        self._time_to_reach *= self._num_lane_changes
        if self._adversary_type is False:
            self._walker_yaw = orientation_yaw
            self._other_actor_target_velocity = 3 + (0.4 * self._num_lane_changes) + 10 # jiwoo 언제 뛰쳐나가는지
            walker_ids = ['walker.pedestrian.0001', 'walker.pedestrian.0002', 'walker.pedestrian.0003', 'walker.pedestrian.0004', 'walker.pedestrian.0005', 'walker.pedestrian.0006', 'walker.pedestrian.0007', 'walker.pedestrian.0008', 'walker.pedestrian.0015', 'walker.pedestrian.0016', 'walker.pedestrian.0017', 'walker.pedestrian.0018', 'walker.pedestrian.0019', 'walker.pedestrian.0020']
            walker = CarlaDataProvider.request_new_actor(random.choice(walker_ids), transform)
            adversary = walker
        else:
            self._other_actor_target_velocity = self._other_actor_target_velocity * self._num_lane_changes
            first_vehicle = CarlaDataProvider.request_new_actor('vehicle.diamondback.century', transform)
            first_vehicle.set_simulate_physics(enabled=False)
            adversary = first_vehicle
        return adversary

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        # cyclist transform
        _start_distance = 0   #jiwoo self.transform이 생성되는 위치와 egovehicle 사이의 거리
        # We start by getting and waypoint in the closest sidewalk.
        waypoint = self._reference_waypoint
        while True:
            wp_next = waypoint.get_right_lane()
            self._num_lane_changes += 1
            if wp_next is None or wp_next.lane_type == carla.LaneType.Sidewalk:
                break
            elif wp_next.lane_type == carla.LaneType.Shoulder:
                # Filter Parkings considered as Shoulders
                if wp_next.lane_width > 2:
                    _start_distance += 1.5
                    waypoint = wp_next
                break
            else:
                _start_distance += 1.5
                waypoint = wp_next

        while True:  # We keep trying to spawn avoiding props

            try:
                self.transform, orientation_yaw = self._calculate_base_transform(_start_distance, waypoint)
                logging.info("transform of walker is "+str(self.transform))
                first_vehicle = self._spawn_adversary(self.transform, orientation_yaw)
                break
            except RuntimeError as r:
                # We keep retrying until we spawn
                print("Base transform is blocking objects ", self.transform)
                _start_distance += 0.4
                self._spawn_attempted += 1
                if self._spawn_attempted >= self._number_of_attempts:
                    raise r

        # Now that we found a possible position we just put the vehicle to the underground
        disp_transform = carla.Transform(
            carla.Location(self.transform.location.x,
                           self.transform.location.y,
                           self.transform.location.z - 500),
            self.transform.rotation)


        first_vehicle.set_transform(disp_transform)
        first_vehicle.set_simulate_physics(enabled=False)
        self.other_actors.append(first_vehicle)

    def _create_behavior(self):
        """
        After invoking this scenario, cyclist will wait for the user
        controlled vehicle to enter trigger distance region,
        the cyclist starts crossing the road once the condition meets,
        then after 60 seconds, a timeout stops the scenario
        """

        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="OccludedObjectCrossing")
        lane_width = self._reference_waypoint.lane_width
        lane_width = lane_width + (1.10 * lane_width * self._num_lane_changes)

        dist_to_trigger = 28 + self._num_lane_changes #+ self._num_lane_changes
        # leaf nodes
        if self._ego_route is not None:
            start_condition = InTriggerDistanceToLocationAlongRoute(self.ego_vehicles[0],
                                                                    self._ego_route,
                                                                    self._reference_waypoint.transform.location,    #jiwoo very important!
                                                                    dist_to_trigger)
        else:
            start_condition = InTimeToArrivalToVehicle(self.ego_vehicles[0],
                                                       self.other_actors[0],
                                                       self._time_to_reach)

        actor_velocity = KeepVelocity(self.other_actors[0],
                                      self._other_actor_target_velocity,
                                      name="walker velocity")
        actor_drive = DriveDistance(self.other_actors[0],
                                    0.5 * lane_width,
                                    name="walker drive distance")
        actor_start_cross_lane = AccelerateToVelocity(self.other_actors[0],
                                                      1.0,
                                                      self._other_actor_target_velocity,
                                                      name="walker crossing lane accelerate velocity")
        actor_cross_lane = DriveDistance(self.other_actors[0],
                                         0.8 * lane_width,
                                         name="walker drive distance for lane crossing ")
        actor_stop_crossed_lane = StopVehicle(self.other_actors[0],
                                              self._other_actor_max_brake,
                                              name="walker stop")
        ego_pass_machine = DriveDistance(self.ego_vehicles[0],
                                         5,
                                         name="ego vehicle passed prop")
        actor_remove = ActorDestroy(self.other_actors[0],
                                    name="Destroying walker")
        end_condition = DriveDistance(self.ego_vehicles[0],
                                      self._ego_vehicle_distance_driven,
                                      name="End condition ego drive distance")

        # non leaf nodes

        scenario_sequence = py_trees.composites.Sequence()
        keep_velocity_other = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="keep velocity other")
        keep_velocity = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="keep velocity")

        # building tree

        root.add_child(scenario_sequence)
        scenario_sequence.add_child(LoggingProgress("Left Walker Crossing Scenario"))
        scenario_sequence.add_child(ActorTransformSetter(self.other_actors[0], self.transform,
                                                         name='TransformSetterTS3walker'))
        scenario_sequence.add_child(HandBrakeVehicle(self.other_actors[0], True))
        scenario_sequence.add_child(start_condition)
        scenario_sequence.add_child(HandBrakeVehicle(self.other_actors[0], False))
        scenario_sequence.add_child(LoggingProgress("walker crossed"))
        scenario_sequence.add_child(keep_velocity)
        scenario_sequence.add_child(actor_stop_crossed_lane)
        scenario_sequence.add_child(LoggingProgress("walker stopped"))
        scenario_sequence.add_child(end_condition)

        keep_velocity.add_child(actor_velocity)
        keep_velocity.add_child(actor_drive)
        keep_velocity_other.add_child(actor_start_cross_lane)
        keep_velocity_other.add_child(actor_cross_lane)
        keep_velocity_other.add_child(ego_pass_machine)

        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        collision_criterion = CollisionTest(self.ego_vehicles[0])
        criteria.append(collision_criterion)

        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()

#jiwoo
class NotCrossing(BasicScenario):

    """
    This class holds everything required for a simple object crash
    without prior vehicle action involving a vehicle and a cyclist/pedestrian,
    The ego vehicle is passing through a road,
    And encounters a cyclist/pedestrian crossing the road.

    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False,
                 debug_mode=False, criteria_enable=True, adversary_type=False, timeout=60):
        """
        Setup all relevant parameters and create scenario
        """
        self._wmap = CarlaDataProvider.get_map()

        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)

        # ego vehicle parameters
        self._ego_vehicle_distance_driven = 40
        # other vehicle parameters
        self._other_actor_target_velocity = 5
        self._other_actor_max_brake = 1.0
        self._time_to_reach = 10
        self._adversary_type = adversary_type  # flag to select either pedestrian (False) or cyclist (True)
        self._walker_yaw = 0
        self._num_lane_changes = 1
        self.transform = None
        self.timeout = timeout
        self._trigger_location = config.trigger_points[0].location
        # Total Number of attempts to relocate a vehicle before spawning
        self._number_of_attempts = 20
        # Number of attempts made so far
        self._spawn_attempted = 0

        self._ego_route = CarlaDataProvider.get_ego_vehicle_route()

        super(NotCrossing, self).__init__("NotCrossing",
                                                    ego_vehicles,
                                                    config,
                                                    world,
                                                    debug_mode,
                                                    criteria_enable=criteria_enable)

    def _calculate_base_transform(self, _start_distance, waypoint):

        lane_width = waypoint.lane_width

        # Patches false junctions
        if self._reference_waypoint.is_junction:
            stop_at_junction = False
        else:
            stop_at_junction = True

        location, _ = get_location_in_distance_from_wp(waypoint, _start_distance, stop_at_junction)
        waypoint = self._wmap.get_waypoint(location)
        offset = {"orientation": 270, "position": 90, "z": 0.6, "k": 1.0}
        position_yaw = waypoint.transform.rotation.yaw + offset['position']
        orientation_yaw = waypoint.transform.rotation.yaw + offset['orientation']
        offset_location = carla.Location(
            offset['k'] * lane_width * 1.1 * math.cos(math.radians(position_yaw)), # default * 1.1
            offset['k'] * lane_width * 1.1 * math.sin(math.radians(position_yaw)))
        location += offset_location
        location.z = self._trigger_location.z + offset['z']
        return carla.Transform(location, carla.Rotation(yaw=orientation_yaw)), orientation_yaw

    def _spawn_adversary(self, transform, orientation_yaw):

        self._time_to_reach *= self._num_lane_changes
        if self._adversary_type is False:
            self._walker_yaw = orientation_yaw
            self._other_actor_target_velocity = 3 + (0.4 * self._num_lane_changes) + 10 # jiwoo 언제 뛰쳐나가는지
            walker_ids = ['walker.pedestrian.0001', 'walker.pedestrian.0002', 'walker.pedestrian.0003', 'walker.pedestrian.0004', 'walker.pedestrian.0005', 'walker.pedestrian.0006', 'walker.pedestrian.0007', 'walker.pedestrian.0008', 'walker.pedestrian.0015', 'walker.pedestrian.0016', 'walker.pedestrian.0017', 'walker.pedestrian.0018', 'walker.pedestrian.0019', 'walker.pedestrian.0020']
            walker = CarlaDataProvider.request_new_actor(random.choice(walker_ids), transform)
            adversary = walker
        else:
            self._other_actor_target_velocity = self._other_actor_target_velocity * self._num_lane_changes
            first_vehicle = CarlaDataProvider.request_new_actor('vehicle.diamondback.century', transform)
            first_vehicle.set_simulate_physics(enabled=False)
            adversary = first_vehicle

        return adversary

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        # cyclist transform
        _start_distance = 12   #jiwoo blocker가 생성되는 위치와 egovehicle 사이의 거리
        # We start by getting and waypoint in the closest sidewalk.
        waypoint = self._reference_waypoint
        while True:
            wp_next = waypoint.get_right_lane()
            self._num_lane_changes += 1
            if wp_next is None or wp_next.lane_type == carla.LaneType.Sidewalk:
                break
            elif wp_next.lane_type == carla.LaneType.Shoulder:
                # Filter Parkings considered as Shoulders
                if wp_next.lane_width > 2:
                    _start_distance += 1.5
                    waypoint = wp_next
                break
            else:
                _start_distance += 1.5
                waypoint = wp_next

        while True:  # We keep trying to spawn avoiding props

            try:
                self.transform, orientation_yaw = self._calculate_base_transform(_start_distance, waypoint)
                logging.info("transform of non crossing walker is "+str(self.transform))
                first_vehicle = self._spawn_adversary(self.transform, orientation_yaw)

                break
            except RuntimeError as r:
                # We keep retrying until we spawn
                print("Base transform is blocking objects ", self.transform)
                _start_distance += 0.4
                self._spawn_attempted += 1
                if self._spawn_attempted >= self._number_of_attempts:
                    raise r

        # Now that we found a possible position we just put the vehicle to the underground
        disp_transform = carla.Transform(
            carla.Location(self.transform.location.x,
                           self.transform.location.y,
                           self.transform.location.z - 500),
            self.transform.rotation)

        first_vehicle.set_transform(disp_transform)
        first_vehicle.set_simulate_physics(enabled=False)
        self.other_actors.append(first_vehicle)

    def _create_behavior(self):
        """
        After invoking this scenario, cyclist will wait for the user
        controlled vehicle to enter trigger distance region,
        the cyclist starts crossing the road once the condition meets,
        then after 60 seconds, a timeout stops the scenario
        """

        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="OccludedObjectCrossing")
        lane_width = self._reference_waypoint.lane_width
        lane_width = lane_width + (1.10 * lane_width * self._num_lane_changes)

        dist_to_trigger = 16 + self._num_lane_changes
        # leaf nodes
        if self._ego_route is not None:
            start_condition = InTriggerDistanceToLocationAlongRoute(self.ego_vehicles[0],
                                                                    self._ego_route,
                                                                    self.transform.location,
                                                                    dist_to_trigger)
        else:
            start_condition = InTimeToArrivalToVehicle(self.ego_vehicles[0],
                                                       self.other_actors[0],
                                                       self._time_to_reach)

        actor_velocity = KeepVelocity(self.other_actors[0],
                                      self._other_actor_target_velocity,
                                      name="walker velocity")
        actor_drive = DriveDistance(self.other_actors[0],
                                    0.3 * lane_width,
                                    name="walker drive distance")
        actor_start_cross_lane = AccelerateToVelocity(self.other_actors[0],
                                                      1.0,
                                                      self._other_actor_target_velocity,
                                                      name="walker crossing lane accelerate velocity")
        actor_cross_lane = DriveDistance(self.other_actors[0],
                                         0.8 * lane_width,
                                         name="walker drive distance for lane crossing ")
        actor_stop_crossed_lane = StopVehicle(self.other_actors[0],
                                              self._other_actor_max_brake,
                                              name="walker stop")
        ego_pass_machine = DriveDistance(self.ego_vehicles[0],
                                         5,
                                         name="ego vehicle passed prop")
        actor_remove = ActorDestroy(self.other_actors[0],
                                    name="Destroying walker")
        end_condition = DriveDistance(self.ego_vehicles[0],
                                      self._ego_vehicle_distance_driven,
                                      name="End condition ego drive distance")

        # non leaf nodes

        scenario_sequence = py_trees.composites.Sequence()
        keep_velocity_other = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="keep velocity other")
        keep_velocity = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="keep velocity")

        # building tree

        root.add_child(scenario_sequence)
        scenario_sequence.add_child(ActorTransformSetter(self.other_actors[0], self.transform,
                                                         name='TransformSetterTS3walker'))
        scenario_sequence.add_child(HandBrakeVehicle(self.other_actors[0], True))

        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        collision_criterion = CollisionTest(self.ego_vehicles[0])
        criteria.append(collision_criterion)

        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()

#jiwoo
class NotCrossingLeft(BasicScenario):

    """
    This class holds everything required for a simple object crash
    without prior vehicle action involving a vehicle and a cyclist/pedestrian,
    The ego vehicle is passing through a road,
    And encounters a cyclist/pedestrian crossing the road.

    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False,
                 debug_mode=False, criteria_enable=True, adversary_type=False, timeout=60):
        """
        Setup all relevant parameters and create scenario
        """
        self._wmap = CarlaDataProvider.get_map()

        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)

        # ego vehicle parameters
        self._ego_vehicle_distance_driven = 40
        # other vehicle parameters
        self._other_actor_target_velocity = 5
        self._other_actor_max_brake = 1.0
        self._time_to_reach = 10
        self._adversary_type = adversary_type  # flag to select either pedestrian (False) or cyclist (True)
        self._walker_yaw = 0
        self._num_lane_changes = 1
        self.transform = None
        self.timeout = timeout
        self._trigger_location = config.trigger_points[0].location
        # Total Number of attempts to relocate a vehicle before spawning
        self._number_of_attempts = 20
        # Number of attempts made so far
        self._spawn_attempted = 0

        self._ego_route = CarlaDataProvider.get_ego_vehicle_route()

        super(NotCrossingLeft, self).__init__("NotCrossingLeft",
                                                    ego_vehicles,
                                                    config,
                                                    world,
                                                    debug_mode,
                                                    criteria_enable=criteria_enable)

    def _calculate_base_transform(self, _start_distance, waypoint):

        lane_width = waypoint.lane_width

        # Patches false junctions
        if self._reference_waypoint.is_junction:
            stop_at_junction = False
        else:
            stop_at_junction = True

        location, _ = get_location_in_distance_from_wp(waypoint, _start_distance, stop_at_junction)
        waypoint = self._wmap.get_waypoint(location)
        offset = {"orientation": -270, "position": -90, "z": 0.6, "k": 1.0}
        position_yaw = waypoint.transform.rotation.yaw + offset['position']
        orientation_yaw = waypoint.transform.rotation.yaw + offset['orientation']
        offset_location = carla.Location(
            offset['k'] * lane_width * 2.0 * math.cos(math.radians(position_yaw)), #jiwoo * modified
            offset['k'] * lane_width * 2.0 * math.sin(math.radians(position_yaw)))
        location += offset_location
        location.z = self._trigger_location.z + offset['z']
        return carla.Transform(location, carla.Rotation(yaw=orientation_yaw)), orientation_yaw

    def _spawn_adversary(self, transform, orientation_yaw):

        self._time_to_reach *= self._num_lane_changes
        if self._adversary_type is False:
            self._walker_yaw = orientation_yaw
            self._other_actor_target_velocity = 3 + (0.4 * self._num_lane_changes) + 10 # jiwoo 언제 뛰쳐나가는지
            walker_ids = ['walker.pedestrian.0001', 'walker.pedestrian.0002', 'walker.pedestrian.0003', 'walker.pedestrian.0004', 'walker.pedestrian.0005', 'walker.pedestrian.0006', 'walker.pedestrian.0007', 'walker.pedestrian.0008', 'walker.pedestrian.0015', 'walker.pedestrian.0016', 'walker.pedestrian.0017', 'walker.pedestrian.0018', 'walker.pedestrian.0019', 'walker.pedestrian.0020']
            walker = CarlaDataProvider.request_new_actor(random.choice(walker_ids), transform)
            adversary = walker
        else:
            self._other_actor_target_velocity = self._other_actor_target_velocity * self._num_lane_changes
            first_vehicle = CarlaDataProvider.request_new_actor('vehicle.diamondback.century', transform)
            first_vehicle.set_simulate_physics(enabled=False)
            adversary = first_vehicle

        return adversary

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        # cyclist transform
        _start_distance = 12   #jiwoo blocker가 생성되는 위치와 egovehicle 사이의 거리
        # We start by getting and waypoint in the closest sidewalk.
        waypoint = self._reference_waypoint
        while True:
            wp_next = waypoint.get_right_lane()
            self._num_lane_changes += 1
            if wp_next is None or wp_next.lane_type == carla.LaneType.Sidewalk:
                break
            elif wp_next.lane_type == carla.LaneType.Shoulder:
                # Filter Parkings considered as Shoulders
                if wp_next.lane_width > 2:
                    _start_distance += 1.5
                    waypoint = wp_next
                break
            else:
                _start_distance += 1.5
                waypoint = wp_next

        while True:  # We keep trying to spawn avoiding props

            try:
                self.transform, orientation_yaw = self._calculate_base_transform(_start_distance, waypoint)
                logging.info("transform of non crossing walker is "+str(self.transform))
                first_vehicle = self._spawn_adversary(self.transform, orientation_yaw)

                break
            except RuntimeError as r:
                # We keep retrying until we spawn
                print("Base transform is blocking objects ", self.transform)
                _start_distance += 0.4
                self._spawn_attempted += 1
                if self._spawn_attempted >= self._number_of_attempts:
                    raise r

        # Now that we found a possible position we just put the vehicle to the underground
        disp_transform = carla.Transform(
            carla.Location(self.transform.location.x,
                           self.transform.location.y,
                           self.transform.location.z - 500),
            self.transform.rotation)

        first_vehicle.set_transform(disp_transform)
        first_vehicle.set_simulate_physics(enabled=False)
        self.other_actors.append(first_vehicle)

    def _create_behavior(self):
        """
        After invoking this scenario, cyclist will wait for the user
        controlled vehicle to enter trigger distance region,
        the cyclist starts crossing the road once the condition meets,
        then after 60 seconds, a timeout stops the scenario
        """

        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="OccludedObjectCrossing")
        lane_width = self._reference_waypoint.lane_width
        lane_width = lane_width + (1.10 * lane_width * self._num_lane_changes)

        dist_to_trigger = 16 + self._num_lane_changes
        # leaf nodes
        if self._ego_route is not None:
            start_condition = InTriggerDistanceToLocationAlongRoute(self.ego_vehicles[0],
                                                                    self._ego_route,
                                                                    self.transform.location,
                                                                    dist_to_trigger)
        else:
            start_condition = InTimeToArrivalToVehicle(self.ego_vehicles[0],
                                                       self.other_actors[0],
                                                       self._time_to_reach)

        actor_velocity = KeepVelocity(self.other_actors[0],
                                      self._other_actor_target_velocity,
                                      name="walker velocity")
        actor_drive = DriveDistance(self.other_actors[0],
                                    0.3 * lane_width,
                                    name="walker drive distance")
        actor_start_cross_lane = AccelerateToVelocity(self.other_actors[0],
                                                      1.0,
                                                      self._other_actor_target_velocity,
                                                      name="walker crossing lane accelerate velocity")
        actor_cross_lane = DriveDistance(self.other_actors[0],
                                         0.8 * lane_width,
                                         name="walker drive distance for lane crossing ")
        actor_stop_crossed_lane = StopVehicle(self.other_actors[0],
                                              self._other_actor_max_brake,
                                              name="walker stop")
        ego_pass_machine = DriveDistance(self.ego_vehicles[0],
                                         5,
                                         name="ego vehicle passed prop")
        actor_remove = ActorDestroy(self.other_actors[0],
                                    name="Destroying walker")
        end_condition = DriveDistance(self.ego_vehicles[0],
                                      self._ego_vehicle_distance_driven,
                                      name="End condition ego drive distance")

        # non leaf nodes

        scenario_sequence = py_trees.composites.Sequence()
        keep_velocity_other = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="keep velocity other")
        keep_velocity = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="keep velocity")

        # building tree

        root.add_child(scenario_sequence)
        scenario_sequence.add_child(ActorTransformSetter(self.other_actors[0], self.transform,
                                                         name='TransformSetterTS3walker'))
        scenario_sequence.add_child(HandBrakeVehicle(self.other_actors[0], True))

        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        collision_criterion = CollisionTest(self.ego_vehicles[0])
        criteria.append(collision_criterion)

        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
