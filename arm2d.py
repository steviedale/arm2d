import gym
from gym.envs.classic_control import rendering
from shapely.geometry import Polygon
import numpy as np


DEBUG = False


class Arm2d(gym.Env):

    def __init__(self):
        self.SCREEN_WIDTH = 600
        self.SCREEN_HEIGHT = self.SCREEN_WIDTH
        self.NUM_ARMS = 6
        self.ARM_LENGTH = (self.SCREEN_HEIGHT/2) / self.NUM_ARMS
        self.ARM_WIDTH = self.ARM_LENGTH / 10
        self.JOINT_ROT_LIMIT = (3/4) * np.pi
        self.CIRCLE_RADIUS = 5
        self.BASE_COLOR = (0, 0, 0)
        self.END_EFFECTOR_COLOR = (0, 0, 0)
        self.TARGET_COLOR = (0, 1, 0)
        self.ARM_COLORS = [
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 0)
        ]
        self.MOVEMENT_COST = 0.1
        self.FLAT_COST = 0.1
        self.TARGET_TOLERANCE = 2

        self.viewer = None
        self.arm_polygons = None
        self.joint_angles = None
        self.end_effect_position = None
        self.target_position = None

    def _check_joint_limits(self, action):
        assert(len(action) == self.NUM_ARMS)
        for joint_movement, joint_angle in zip(action, self.joint_angles):
            if abs(joint_angle + joint_movement) > self.JOINT_ROT_LIMIT:
                return False
            return True

    def _update_arm_polygons(self, joint_angle_inc):
        angle = 0
        position = (0, 0)
        for i in range(self.NUM_ARMS):
            points = [None, None, None, None]
            angle += self.joint_angles[i] + joint_angle_inc[i]
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
                position = (position[0], position[1] + self.ARM_LENGTH)
            elif angle == np.pi or angle == -np.pi:
                points[0] = (position[0] + self.ARM_WIDTH/2, position[1])
                points[1] = (position[0] + self.ARM_WIDTH/2, position[1] - self.ARM_LENGTH)
                points[2] = (position[0] - self.ARM_WIDTH/2, position[1] - self.ARM_LENGTH)
                points[3] = (position[0] - self.ARM_WIDTH/2, position[1])
                position = (position[0], position[1] - self.ARM_LENGTH)
            elif angle == np.pi/2:
                points[0] = (position[0], position[1] + self.ARM_WIDTH/2)
                points[1] = (position[0] + self.ARM_LENGTH, position[1] + self.ARM_WIDTH/2)
                points[2] = (position[0] + self.ARM_LENGTH, position[1] - self.ARM_WIDTH/2)
                points[3] = (position[0], position[1] - self.ARM_WIDTH/2)
                position = (position[0] + self.ARM_LENGTH, position[1])
            elif angle == -np.pi/2:
                points[0] = (position[0], position[1] - self.ARM_WIDTH/2)
                points[1] = (position[0] - self.ARM_LENGTH, position[1] - self.ARM_WIDTH/2)
                points[2] = (position[0] - self.ARM_LENGTH, position[1] + self.ARM_WIDTH/2)
                points[3] = (position[0], position[1] + self.ARM_WIDTH/2)
                position = (position[0] - self.ARM_LENGTH, position[1])
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
                position = (position[0] + x_l, position[1] + y_l)
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
                position = (position[0] + x_l, position[1] + y_l)
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
                position = (position[0] + x_l, position[1] + y_l)
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
                position = (position[0] + x_l, position[1] + y_l)
            else:
                # this shouldn't happen
                assert(False)
            assert(None not in points)
            self.arm_polygons[i] = Polygon(points)
            self.joint_angles[i] += joint_angle_inc[i]
        self.end_effect_position = position

    def _check_arm_collisions(self):
        collision = False
        for i in range(self.NUM_ARMS-2):
            for j in range(i+2, self.NUM_ARMS):
                if self.arm_polygons[i].intersects(self.arm_polygons[j]):
                    if DEBUG:
                        print("collision of ARM_{} and ARM_{}".format(i, j))
                    collision = True
        return collision

    def _target_reached(self):
        return (abs(self.end_effect_position[0] - self.target_position[0]) < self.TARGET_TOLERANCE and
                abs(self.end_effect_position[1] - self.target_position[1]) < self.TARGET_TOLERANCE)

    def reset(self):
        self.arm_polygons = []
        self.joint_angles = []
        self.end_effect_position = [0, self.ARM_LENGTH * self.NUM_ARMS]
        self.target_position = (100, 200)

        for i in range(self.NUM_ARMS):
            points = [
                (-self.ARM_WIDTH/2, i*self.ARM_LENGTH),
                (-self.ARM_WIDTH/2, (i+1)*self.ARM_LENGTH),
                (self.ARM_WIDTH/2, (i+1)*self.ARM_LENGTH),
                (self.ARM_WIDTH/2, i*self.ARM_LENGTH)
            ]
            polygon = Polygon(points)
            self.arm_polygons.append(polygon)
            self.joint_angles.append(0)

    def _get_state(self):
        return self.joint_angles

    def step(self, action):
        assert(self._check_joint_limits(action))
        self._update_arm_polygons(action)
        state = self._get_state()

        if self._check_arm_collisions():
            return state, -100, True, "self-collision"

        if self._target_reached():
            return state, 100, True, "target reached"

        reward = sum(action) * self.MOVEMENT_COST + self.FLAT_COST
        return state, reward, False, ""

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
        t = rendering.Transform(translation=(self.SCREEN_WIDTH/2 + self.end_effect_position[0],
                                             self.SCREEN_HEIGHT/2 + self.end_effect_position[1]))
        self.viewer.draw_circle(self.CIRCLE_RADIUS, color=self.END_EFFECTOR_COLOR).add_attr(t)
        # draw base
        t = rendering.Transform(translation=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
        self.viewer.draw_circle(self.CIRCLE_RADIUS, color=self.BASE_COLOR).add_attr(t)
        # draw target
        t = rendering.Transform(translation=(self.SCREEN_WIDTH/2 + self.target_position[0],
                                             self.SCREEN_HEIGHT/2 + self.target_position[1]))
        self.viewer.draw_circle(self.CIRCLE_RADIUS, color=self.TARGET_COLOR).add_attr(t)
        return self.viewer.render(return_rgb_array='rgb_array')


if __name__ == '__main__':
    import time
    agent = Arm2d()
    agent.reset()
    agent.render()
    while True:
        inc = np.pi/1000
        agent.step([-inc, -inc, -inc, -inc, -inc, -inc])
        agent.render()
        time.sleep(0.005)