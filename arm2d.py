import gym
from gym.envs.classic_control import rendering
from shapely.geometry import Polygon
import numpy as np


class Arm2d(gym.Env):

    def __init__(self):
        self.SCREEN_WIDTH = 600
        self.SCREEN_HEIGHT = 600
        self.NUM_ARMS = 3
        self.ARM_COLORS = [
            (0, 0, 1),
            (0, 1, 0),
            (1, 0, 0)
        ]
        self.ARM_LENGTH = 100
        self.ARM_WIDTH = 10
        self.JOINT_ROT_LIMIT = (3/4) * np.pi
        self.CIRCLE_RADIUS = 5
        self.CIRCLE_COLOR = (0, 0, 0)

        self.viewer = None
        self.arm_polygons = None
        self.joint_angles = None
        self.end_effect_position = None

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

    def reset(self):
        self.arm_polygons = []
        self.joint_angles = []
        self.end_effect_position = [0, self.ARM_LENGTH * self.NUM_ARMS]

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

    def step(self, action):
        assert(self._check_joint_limits(action))
        self._update_arm_polygons(action)

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = rendering.Viewer(self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
            self.viewer.set_bounds(0, self.SCREEN_WIDTH, 0, self.SCREEN_HEIGHT)
        for i, polygon in enumerate(self.arm_polygons):
            x_list, y_list = polygon.exterior.coords.xy
            points = [(x+self.SCREEN_WIDTH/2, y+self.SCREEN_HEIGHT/2) for x, y in zip(x_list, y_list)]
            self.viewer.draw_polygon(points, color=self.ARM_COLORS[i])
        # draw end effector
        t = rendering.Transform(translation=(self.SCREEN_WIDTH/2 + self.end_effect_position[0],
                                             self.SCREEN_HEIGHT/2 + self.end_effect_position[1]))
        self.viewer.draw_circle(self.CIRCLE_RADIUS, color=self.CIRCLE_COLOR).add_attr(t)
        # draw base
        t = rendering.Transform(translation=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
        self.viewer.draw_circle(self.CIRCLE_RADIUS, color=self.CIRCLE_COLOR).add_attr(t)
        return self.viewer.render(return_rgb_array='rgb_array')


if __name__ == '__main__':
    import time
    agent = Arm2d()
    agent.reset()
    agent.render()
    while True:
        inc = np.pi/100
        agent.step([inc, -inc, inc])
        agent.render()
        time.sleep(0.1)
    print('done')