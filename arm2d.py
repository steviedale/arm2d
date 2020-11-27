import numpy as np

import gym
from gym.envs.classic_control import rendering
from shapely.geometry import Polygon
import time


DEBUG = False


class Arm2d(gym.Env):

    def __init__(self, has_obstacles=True):
        # MEASUREMENTS
        self.SCREEN_WIDTH = 600
        self.SCREEN_HEIGHT = self.SCREEN_WIDTH
        self.NUM_ARMS = 6
        self.ARM_LENGTH = (self.SCREEN_HEIGHT/2) / self.NUM_ARMS
        self.ARM_WIDTH = self.ARM_LENGTH / 10
        self.JOINT_LIMIT = (7 / 8) * np.pi
        self.INTERPOLATE_INC = (1 / 100) * np.pi
        self.MAX_JOINT_ROTATION = (1 / 10) * np.pi
        self.CIRCLE_RADIUS = 5
        self.TARGET_RADIUS = 10
        # CONSTRAINTS
        self.MIN_CENTER_TO_TARGET_DISTANCE = self.SCREEN_WIDTH / 8
        self.MAX_CENTER_TO_TARGET_DISTANCE = self.SCREEN_WIDTH / 2
        self.MIN_END_EFFECTOR_TO_TARGET_DISTANCE = self.SCREEN_WIDTH / 8
        self.MAX_END_EFFECTOR_TO_TARGET_DISTANCE = self.SCREEN_WIDTH / 4
        # COLORS
        self.BASE_COLOR = (0, 0, 0)
        self.END_EFFECTOR_COLOR = (0, 0, 0)
        self.TARGET_COLOR = (0, 0.8, 0)
        self.ARM_COLORS = [
            (255 / 255, 51 / 255, 51 / 255),    # red
            (255 / 255, 255 / 255, 51 / 255),   # yellow
            (255 / 255, 51 / 255, 255 / 255),   # pink
            (0 / 255, 204 / 255, 0 / 255),     # green
            (0 / 255, 128 / 255, 255 / 255),    # purple
            (255 / 255, 128 / 255, 0 / 255),    # orange
            (0 / 255, 0 / 255, 255 / 255),      # blue
            (0 / 255, 255 / 255, 128 / 255),    # teal
        ]
        max_norm = np.linalg.norm(np.ones(self.NUM_ARMS)*self.MAX_JOINT_ROTATION)
        # COSTS
        self.MOVEMENT_COST = -0.5 / max_norm
        self.FLAT_COST = -0.1
        self.JOINT_LIMIT_VIOLATION_COST = -2.0 / max_norm
        self.COLLISION_COST = -2.0 / max_norm
        self.OVERSHOT_COST = -100.0 / max_norm
        self.LINEAR_DEVIATION_COST = -1.0 / self.SCREEN_WIDTH
        # REWARDS
        self.TARGET_PROXIMITY_REWARD = 500.0
        self.TARGET_REACHED_REWARD = 500.0

        self.has_obstacles = has_obstacles

        self.action_space = gym.spaces.Box(-1.0, 1.0, (self.NUM_ARMS,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(self.NUM_ARMS+4,), dtype=np.float32)

        self.viewer = None
        self.arm_polygons = None
        self.joint_angles = None
        self.end_effector_position = None
        self.end_effector_position_record = None
        self.target_position = None
        self.distance_to_target = None
        self.initial_distance_to_target = None
        self.start_position = None

    def _update_arm_pose(self):
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
        while True:
            for i in range(self.NUM_ARMS):
                self.joint_angles[i] = -self.JOINT_LIMIT + np.random.random() * self.JOINT_LIMIT * 2
                # self.joint_angles[i] = -(self.JOINT_ROT_LIMIT/2) + np.random.random() * self.JOINT_ROT_LIMIT
            self._update_arm_pose()
            collision = self._check_arm_collisions()
            dist_to_end_effector = np.linalg.norm(self.end_effector_position)
            if (
                not collision
                and self.MIN_CENTER_TO_TARGET_DISTANCE < dist_to_end_effector < self.MAX_CENTER_TO_TARGET_DISTANCE
            ):
                break
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
        joint_angles = self.joint_angles / self.JOINT_LIMIT
        target_position = self.target_position / (self.SCREEN_WIDTH / 2)
        start_position = self.start_position / (self.SCREEN_WIDTH / 2)
        return np.append(np.append(joint_angles, start_position), target_position)

    def _get_distance_from_line(self):
        rise = self.target_position[1] - self.start_position[1]
        run = self.target_position[0] - self.start_position[0]
        # vertical target trajectory
        if run == 0:
            return np.abs(self.end_effector_position[0] - self.target_position[0])
        # horizontal target trajectory
        elif rise == 0:
            return np.abs(self.end_effector_position[1] - self.target_position[1])
        else:
            m1 = rise/run
            # b = y - mx
            b1 = self.target_position[1] - m1 * self.target_position[0]
            m2 = -1/m1
            # b = y - mx
            b2 = self.end_effector_position[1] - m2 * self.end_effector_position[0]
            # m1 * x + b1 = m2 * x + b2
            # m1 * x - m2 * x = b2 - b1
            # (m1 - m2) * x = b2 - b1
            x = (b2 - b1) / (m1 - m2)
            y = m1 * x + b1
            p = np.array([x, y])
            return np.linalg.norm(self.end_effector_position - p)

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, random_arm_position=True, random_target_position=True, arm_start_position=None,
              target_position=None):
        # ensure parameters are contradictory
        if random_arm_position and arm_start_position:
            raise Exception("random_arm_position and arm_start_position are mutually exclusive")
        if random_target_position and target_position:
            raise Exception("random_target_position and target_start_position are mutually exclusive")

        self.end_effector_position_record = []
        self.arm_polygons = [None] * self.NUM_ARMS
        self.joint_angles = np.zeros(self.NUM_ARMS)
        self.end_effector_position = np.zeros(2)
        self.target_position = np.ones(2) * 100.0

        if arm_start_position is not None:
            self.joint_angles = arm_start_position

        if random_arm_position:
            self._set_random_robot_position()

        self._update_arm_pose()
        self.start_position = self.end_effector_position
        self.end_effector_position_record.append(self.start_position)

        if target_position is not None:
            self.target_position = target_position

        if random_target_position:
            while True:
                self.target_position = np.random.rand(2) * self.SCREEN_WIDTH - self.SCREEN_WIDTH/2
                distance_from_center = np.linalg.norm(self.target_position)
                distance_from_end_effector = np.linalg.norm(self.target_position - self.end_effector_position)
                if (
                        self.MIN_CENTER_TO_TARGET_DISTANCE < distance_from_center < self.MAX_CENTER_TO_TARGET_DISTANCE and
                        self.MIN_END_EFFECTOR_TO_TARGET_DISTANCE < distance_from_end_effector < self.MAX_END_EFFECTOR_TO_TARGET_DISTANCE and
                        not self._target_reached()
                ):
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
                self._update_arm_pose()

                if continuous_render:
                    self.render()
                    time.sleep(0.01)

                if self._check_arm_collisions():
                    # reverse collision
                    self.joint_angles -= joint_inc
                    self._update_arm_pose()
                    reward += self.COLLISION_COST * np.linalg.norm(joint_inc * (num_steps - step))
                    info['violations'].append('arm_self_collision')
                    break

                if self._joint_limit_violated():
                    # reverse limit violation
                    self.joint_angles -= joint_inc
                    self._update_arm_pose()
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

                # add cost of linear deviation
                dist = self._get_distance_from_line()
                reward += self.LINEAR_DEVIATION_COST * dist
                self.end_effector_position_record.append(self.end_effector_position)

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
        # draw target path
        self.viewer.draw_polyline(
            (
                (self.SCREEN_WIDTH/2 + self.start_position[0], self.SCREEN_WIDTH/2 + self.start_position[1]),
                (self.SCREEN_WIDTH/2 + self.target_position[0], self.SCREEN_WIDTH/2 + self.target_position[1]),
            ),
            linewidth=5,
            color = (0.7, 0.7, 0.7)
        )
        # draw trajectory
        path = [(self.SCREEN_WIDTH/2 + p[0], self.SCREEN_WIDTH/2 + p[1]) for p in self.end_effector_position_record]
        self.viewer.draw_polyline(
            path,
            linewidth=5,
            color=(0, 1, 0)
        )
        return self.viewer.render(return_rgb_array='rgb_array')


if __name__ == '__main__':
    agent = Arm2d()
    agent.reset(random_arm_position=False)
    agent.render()
    done = False
    start_time = time.time()
    while not done:
        inc = np.pi/8
        # next_state, reward, done, info = agent.step([-inc, -inc, -inc, -inc, -inc, -inc])
        next_state, reward, done, info = agent.step([inc, 0, 0, 0, 0, 0])
        agent.render()
        # time.sleep(0.1)
    end_time = time.time()
    print("duration: {}".format(round(end_time - start_time, 2)))