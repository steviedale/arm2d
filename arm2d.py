import numpy as np

import gym
from gym.envs.classic_control import rendering
from shapely.geometry import Polygon
import time


DEBUG = False


class Arm2d(gym.Env):

    def __init__(self):
        # MEASUREMENTS
        self.SCREEN_WIDTH = 600
        self.SCREEN_HEIGHT = self.SCREEN_WIDTH
        self.NUM_ARMS = 4
        self.ARM_LENGTH = (self.SCREEN_HEIGHT/2) / self.NUM_ARMS
        self.ARM_WIDTH = self.ARM_LENGTH / 10
        self.JOINT_LIMIT = (7 / 8) * np.pi
        self.INTERPOLATE_INC = (1 / 100) * np.pi
        self.MAX_JOINT_ROTATION = (1 / 10) * np.pi
        self.CIRCLE_RADIUS = 5
        self.TARGET_RADIUS = 10
        # COLORS
        self.BASE_COLOR = (0, 0, 0)
        self.END_EFFECTOR_COLOR = (0, 0, 0)
        self.TARGET_COLOR = (0, 0.8, 0)
        self.ARM_COLORS = [
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 0)
        ]
        # COSTS
        self.MOVEMENT_COST = -0.5 / self.MAX_JOINT_ROTATION
        self.FLAT_COST = -0.5
        self.JOINT_LIMIT_VIOLATION_COST = -2.0 / self.MAX_JOINT_ROTATION
        self.COLLISION_COST = -2.0 / self.MAX_JOINT_ROTATION
        self.OVERSHOT_COST = -2.0 / self.MAX_JOINT_ROTATION
        # REWARDS
        self.TARGET_PROXIMITY_REWARD = 100.0
        self.TARGET_REACHED_REWARD = 100.0

        self.action_space = gym.spaces.Box(-1.0, 1.0, (self.NUM_ARMS,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(self.NUM_ARMS+2,), dtype=np.float32)

        self.viewer = None
        self.arm_polygons = None
        self.joint_angles = None
        self.end_effector_position = None
        self.target_position = None
        self.distance_to_target = None

    def _update_arm_polygons(self):
        angle = 0
        position = np.zeros(2)
        for i in range(self.NUM_ARMS):
            points = [None, None, None, None]
            angle += self.joint_angles[i]
            # this shouldn't happen
            assert(2*np.pi > angle > -2*np.pi)
            # normalize angle
            if angle > np.pi:
                angle = -np.pi + (angle - np.pi)
            elif angle < -np.pi:
                angle = np.pi + (angle + np.pi)

            if angle == 0:
                points[0] = (position[0] - self.ARM_WIDTH/2, position[1])
                points[1] = (position[0] - self.ARM_WIDTH/2, position[1] + self.ARM_LENGTH)
                points[2] = (position[0] + self.ARM_WIDTH/2, position[1] + self.ARM_LENGTH)
                points[3] = (position[0] + self.ARM_WIDTH/2, position[1])
                position[1] += self.ARM_LENGTH
            elif angle == np.pi or angle == -np.pi:
                points[0] = (position[0] + self.ARM_WIDTH/2, position[1])
                points[1] = (position[0] + self.ARM_WIDTH/2, position[1] - self.ARM_LENGTH)
                points[2] = (position[0] - self.ARM_WIDTH/2, position[1] - self.ARM_LENGTH)
                points[3] = (position[0] - self.ARM_WIDTH/2, position[1])
                position[1] -= self.ARM_LENGTH
            elif angle == np.pi/2:
                points[0] = (position[0], position[1] + self.ARM_WIDTH/2)
                points[1] = (position[0] + self.ARM_LENGTH, position[1] + self.ARM_WIDTH/2)
                points[2] = (position[0] + self.ARM_LENGTH, position[1] - self.ARM_WIDTH/2)
                points[3] = (position[0], position[1] - self.ARM_WIDTH/2)
                position[0] += self.ARM_LENGTH
            elif angle == -np.pi/2:
                points[0] = (position[0], position[1] - self.ARM_WIDTH/2)
                points[1] = (position[0] - self.ARM_LENGTH, position[1] - self.ARM_WIDTH/2)
                points[2] = (position[0] - self.ARM_LENGTH, position[1] + self.ARM_WIDTH/2)
                points[3] = (position[0], position[1] + self.ARM_WIDTH/2)
                position[0] -= self.ARM_LENGTH
            elif -np.pi < angle < -np.pi/2:
                theta = -angle - np.pi/2
                x_w = self.ARM_WIDTH/2 * np.sin(theta)
                y_w = -self.ARM_WIDTH/2 * np.cos(theta)
                x_l = -self.ARM_LENGTH * np.cos(theta)
                y_l = -self.ARM_LENGTH * np.sin(theta)
                points[0] = (position[0] + x_w, position[1] + y_w)
                points[1] = (position[0] + x_w + x_l, position[1] + y_w + y_l)
                points[2] = (position[0] - x_w + x_l, position[1] - y_w + y_l)
                points[3] = (position[0] - x_w, position[1] - y_w)
                position += np.array([x_l, y_l])
            elif -np.pi/2 < angle < 0:
                theta = -angle
                x_w = -self.ARM_WIDTH/2 * np.cos(theta)
                y_w = -self.ARM_WIDTH/2 * np.sin(theta)
                x_l = -self.ARM_LENGTH * np.sin(theta)
                y_l = self.ARM_LENGTH * np.cos(theta)
                points[0] = (position[0] + x_w, position[1] + y_w)
                points[1] = (position[0] + x_w + x_l, position[1] + y_w + y_l)
                points[2] = (position[0] - x_w + x_l, position[1] - y_w + y_l)
                points[3] = (position[0] - x_w, position[1] - y_w)
                position += np.array([x_l, y_l])
            elif 0 < angle < np.pi/2:
                theta = angle
                x_w = -self.ARM_WIDTH/2 * np.cos(theta)
                y_w = self.ARM_WIDTH/2 * np.sin(theta)
                x_l = self.ARM_LENGTH * np.sin(theta)
                y_l = self.ARM_LENGTH * np.cos(theta)
                points[0] = (position[0] + x_w, position[1] + y_w)
                points[1] = (position[0] + x_w + x_l, position[1] + y_w + y_l)
                points[2] = (position[0] - x_w + x_l, position[1] - y_w + y_l)
                points[3] = (position[0] - x_w, position[1] - y_w)
                position += np.array([x_l, y_l])
            elif np.pi/2 < angle < np.pi:
                theta = angle - np.pi/2
                x_w = self.ARM_WIDTH/2 * np.sin(theta)
                y_w = self.ARM_WIDTH/2 * np.cos(theta)
                x_l = self.ARM_LENGTH * np.cos(theta)
                y_l = -self.ARM_LENGTH * np.sin(theta)
                points[0] = (position[0] + x_w, position[1] + y_w)
                points[1] = (position[0] + x_w + x_l, position[1] + y_w + y_l)
                points[2] = (position[0] - x_w + x_l, position[1] - y_w + y_l)
                points[3] = (position[0] - x_w, position[1] - y_w)
                position += np.array([x_l, y_l])
            # this shouldn't happen
            else:
                assert(False)
            # neither should this
            assert(None not in points)
            self.arm_polygons[i] = Polygon(points)
        self.end_effector_position = position

    def _set_random_robot_position(self):
        collision = True
        counter = 0
        while collision:
            for i in range(self.NUM_ARMS):
                self.joint_angles[i] = -self.JOINT_LIMIT + np.random.random() * self.JOINT_LIMIT * 2
                # self.joint_angles[i] = -(self.JOINT_ROT_LIMIT/2) + np.random.random() * self.JOINT_ROT_LIMIT
            self._update_arm_polygons()
            collision = self._check_arm_collisions()
            counter += 1
        if DEBUG:
            print("_set_random_robot_position took {} attempts".format(counter))

    def _check_arm_collisions(self):
        collision = False
        for i in range(self.NUM_ARMS-2):
            for j in range(i+2, self.NUM_ARMS):
                if self.arm_polygons[i].intersects(self.arm_polygons[j]):
                    if DEBUG:
                        print("collision of ARM_{} and ARM_{}".format(i, j))
                    collision = True
        return collision

    def _joint_limit_violated(self):
        return np.max(np.abs(self.joint_angles)) > self.JOINT_LIMIT

    def _target_reached(self):
        return np.max(np.abs(self.end_effector_position - self.target_position)) < self.TARGET_RADIUS

    def _get_distance_to_target(self):
        return np.linalg.norm(self.end_effector_position - self.target_position)

    def _update_distance_to_target(self):
        self.distance_to_target = self._get_distance_to_target()

    def _get_state(self):
        j = self.joint_angles / self.JOINT_LIMIT
        p = self.target_position / (self.SCREEN_WIDTH / 2)
        return np.append(j, p)

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, random_arm_position=True, random_target_position=True, arm_start_position=None,
              target_position=None):
        # ensure parameters are contradictory
        if random_arm_position and arm_start_position:
            raise Exception("random_arm_position and arm_start_position are mutually exclusive")
        if random_target_position and target_position:
            raise Exception("random_target_position and target_start_position are mutually exclusive")

        self.arm_polygons = [None] * self.NUM_ARMS
        self.joint_angles = np.zeros(self.NUM_ARMS)
        self.end_effector_position = np.zeros(2)
        self.target_position = np.ones(2) * 100.0

        if arm_start_position is not None:
            self.joint_angles = arm_start_position

        if random_arm_position:
            self._set_random_robot_position()

        self._update_arm_polygons()

        if target_position is not None:
            self.target_position = target_position

        if random_target_position:
            while True:
                self.target_position = np.random.rand(2) * self.SCREEN_WIDTH - self.SCREEN_WIDTH/2
                distance_from_center = np.linalg.norm(self.target_position)
                if distance_from_center < self.SCREEN_WIDTH/2 and not self._target_reached():
                    break

        self._update_distance_to_target()
        self.initial_distance_to_target = self._get_distance_to_target()

        return self._get_state()

    def step(self, action, continuous_render=True):
        action = np.array(action) * self.MAX_JOINT_ROTATION
        info = {'violations': []}

        reward = 0
        # calculate interpolation steps
        max_angle_inc = np.max(np.abs(action))
        num_steps = int(np.ceil(max_angle_inc / self.INTERPOLATE_INC))
        if num_steps > 0:
            joint_inc = action / num_steps
        for step in range(num_steps):
            self.joint_angles += joint_inc
            self._update_arm_polygons()

            if continuous_render:
                self.render()
                time.sleep(0.01)

            if self._check_arm_collisions():
                # reverse collision
                self.joint_angles -= joint_inc
                self._update_arm_polygons()
                reward += self.COLLISION_COST * np.linalg.norm(joint_inc * (num_steps - step))
                info['violations'].append('arm_self_collision')
                break

            if self._joint_limit_violated():
                # reverse limit violation
                self.joint_angles -= joint_inc
                self._update_arm_polygons()
                reward += self.JOINT_LIMIT_VIOLATION_COST * np.linalg.norm(joint_inc * (num_steps - step))
                info['violations'].append('joint_limit_violation')
                break

            if self._target_reached():
                # target reached reward
                reward += self.TARGET_REACHED_REWARD
                # proximity reward
                distance_before = self.distance_to_target
                self._update_distance_to_target()
                reward += self.TARGET_PROXIMITY_REWARD * (
                         distance_before - self.distance_to_target) / self.initial_distance_to_target
                # overshot cost
                reward += self.OVERSHOT_COST * np.linalg.norm(joint_inc * (num_steps - step - 1))
                return self._get_state(), reward, True, "target reached"

        # proximity reward
        distance_before = self.distance_to_target
        self._update_distance_to_target()
        reward += self.TARGET_PROXIMITY_REWARD * (distance_before - self.distance_to_target) / self.initial_distance_to_target
        # movement cost
        reward += sum(action) * self.MOVEMENT_COST
        # step cost (flat cost)
        reward += self.FLAT_COST
        return self._get_state(), reward, False, info

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = rendering.Viewer(self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
            self.viewer.set_bounds(0, self.SCREEN_WIDTH, 0, self.SCREEN_HEIGHT)
        # draw arm
        for i, polygon in enumerate(self.arm_polygons):
            x_list, y_list = polygon.exterior.coords.xy
            points = [(x+self.SCREEN_WIDTH/2, y+self.SCREEN_HEIGHT/2) for x, y in zip(x_list, y_list)]
            self.viewer.draw_polygon(points, color=self.ARM_COLORS[i])
        # draw end effector
        t = rendering.Transform(translation=(self.SCREEN_WIDTH/2 + self.end_effector_position[0],
                                             self.SCREEN_HEIGHT/2 + self.end_effector_position[1]))
        self.viewer.draw_circle(self.CIRCLE_RADIUS, color=self.END_EFFECTOR_COLOR).add_attr(t)
        # draw base
        t = rendering.Transform(translation=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
        self.viewer.draw_circle(self.CIRCLE_RADIUS, color=self.BASE_COLOR).add_attr(t)
        # draw target
        t = rendering.Transform(translation=(self.SCREEN_WIDTH/2 + self.target_position[0],
                                             self.SCREEN_HEIGHT/2 + self.target_position[1]))
        self.viewer.draw_circle(self.TARGET_RADIUS, color=self.TARGET_COLOR).add_attr(t)
        return self.viewer.render(return_rgb_array='rgb_array')

    def seed(self, seed):
        np.random.seed(seed)


if __name__ == '__main__':
    agent = Arm2d()
    agent.reset(random_arm_position=False)
    agent.render()
    done = False
    start_time = time.time()
    while not done:
        inc = np.pi/8
        # next_state, reward, done, info = agent.step([-inc, -inc, -inc, -inc, -inc, -inc])
        next_state, reward, done, info = agent.step([inc, 0, 0, 0, 0, 0], render=True)
        agent.render()
        # time.sleep(0.1)
    end_time = time.time()
    print("duration: {}".format(round(end_time - start_time, 2)))