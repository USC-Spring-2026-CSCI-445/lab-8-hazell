#!/usr/bin/env python3
from typing import Optional, Tuple, List, Dict
from argparse import ArgumentParser
from math import inf, sqrt, atan2, pi
from time import sleep, time
import queue
import json
import math
from random import uniform
import copy

import scipy
import numpy as np
import rospy
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Twist, Point32, PoseStamped, Pose, Vector3, Quaternion, Point, PoseArray
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan, PointCloud, ChannelFloat32
from visualization_msgs.msg import MarkerArray, Marker
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import scipy.stats
from numpy.random import choice

np.set_printoptions(linewidth=200)

# AABB format: (x_min, x_max, y_min, y_max)
OBS_TYPE = Tuple[float, float, float, float]
# Position format: {"x": x, "y": y, "theta": theta}
POSITION_TYPE = Dict[str, float]

# don't change this
GOAL_THRESHOLD = 0.1


def angle_to_0_to_2pi(angle: float) -> float:
    while angle < 0:
        angle += 2 * pi
    while angle > 2 * pi:
        angle -= 2 * pi
    return angle


def angle_to_neg_pi_to_pi(angle: float) -> float:
    while angle < -pi:
        angle += 2 * pi
    while angle > pi:
        angle -= 2 * pi
    return angle


# see https://rootllama.wordpress.com/2014/06/20/ray-line-segment-intersection-test-in-2d/
def ray_line_intersection(ray_origin, ray_direction_rad, point1, point2):
    # Convert to numpy arrays
    ray_origin = np.array(ray_origin, dtype=np.float32)
    ray_direction = np.array([math.cos(ray_direction_rad), math.sin(ray_direction_rad)])
    point1 = np.array(point1, dtype=np.float32)
    point2 = np.array(point2, dtype=np.float32)

    # Ray-Line Segment Intersection Test in 2D
    v1 = ray_origin - point1
    v2 = point2 - point1
    v3 = np.array([-ray_direction[1], ray_direction[0]])
    denominator = np.dot(v2, v3)
    if denominator == 0:
        return None
    t1 = np.cross(v2, v1) / denominator
    t2 = np.dot(v1, v3) / denominator
    if t1 >= 0.0 and 0.0 <= t2 <= 1.0:
        return [ray_origin + t1 * ray_direction]
    return None


class Map:
    def __init__(self, obstacles: List[OBS_TYPE], map_aabb: Tuple):
        self.obstacles = obstacles
        self.map_aabb = map_aabb

    @property
    def top_right(self) -> Tuple[float, float]:
        return self.map_aabb[1], self.map_aabb[3]

    @property
    def bottom_left(self) -> Tuple[float, float]:
        return self.map_aabb[0], self.map_aabb[2]

    def draw_distances(self, origins: List[Tuple[float, float]]):
        """Example usage:
        map_ = Map(obstacles, map_aabb)
        map_.draw_distances([(0.0, 0.0), (3, 3), (1.5, 1.5)])
        """

        # Draw scene
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig, ax = plt.subplots(figsize=(10, 10))
        fig.tight_layout()
        x_min_global, x_max_global, y_min_global, y_max_global = self.map_aabb
        for aabb in self.obstacles:
            width = aabb[1] - aabb[0]
            height = aabb[3] - aabb[2]
            rect = patches.Rectangle(
                (aabb[0], aabb[2]), width, height, linewidth=2, edgecolor="r", facecolor="r", alpha=0.4
            )
            ax.add_patch(rect)
        ax.set_xlim(x_min_global, x_max_global)
        ax.set_ylim(y_min_global, y_max_global)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("2D Plot of Obstacles")
        ax.set_aspect("equal", "box")
        plt.grid(True)

        # Draw rays
        angles = np.linspace(0, 2 * math.pi, 10, endpoint=False)
        for origin in origins:
            for angle in angles:
                closest_distance = self.closest_distance(origin, angle)
                if closest_distance is not None:
                    x = origin[0] + closest_distance * math.cos(angle)
                    y = origin[1] + closest_distance * math.sin(angle)
                    plt.plot([origin[0], x], [origin[1], y], "b-")
        plt.show()

    def closest_distance(self, origin: Tuple[float, float], angle: float) -> Optional[float]:
        """Returns the closest distance to an obstacle from the given origin in the given direction `angle`. If no
        intersection is found, returns `None`.
        """

        def lines_from_obstacle(obstacle: OBS_TYPE):
            """Returns the four lines of the given AABB format obstacle.
            Example usage: `point0, point1 = lines_from_obstacle(self.obstacles[0])`
            """
            x_min, x_max, y_min, y_max = obstacle
            return [
                [(x_min, y_min), (x_max, y_min)],
                [(x_max, y_min), (x_max, y_max)],
                [(x_max, y_max), (x_min, y_max)],
                [(x_min, y_max), (x_min, y_min)],
            ]

        # Iterate over the obstacles in the map to find the closest distance (if there is one). Remember that the
        # obstacles are represented as a list of AABBs (Axis-Aligned Bounding Boxes) with the format
        # (x_min, x_max, y_min, y_max).
        result = None
        origin = np.array(origin)

        for obstacle in self.obstacles:
            for line in lines_from_obstacle(obstacle):
                p = ray_line_intersection(origin, angle, line[0], line[1])
                if p is None:
                    continue

                dist = np.linalg.norm(np.array(p) - origin)
                if result is None:
                    result = dist
                else:
                    result = min(result, dist)
        return result

# PID controller class
######### Your code starts here #########
class PIDController:
    """
    Generates control action taking into account instantaneous error (proportional action),
    accumulated error (integral action) and rate of change of error (derivative action).
    """

    def __init__(self, kP, kI, kD, kS, u_min, u_max):
        assert u_min < u_max, "u_min should be less than u_max"
        # Initialize PID variables here
        ######### Your code starts here #########
        self.kP = kP
        self.kI = kI
        self.kD = kD
        self.kS = kS
        self.u_min = u_min
        self.u_max = u_max
        self.t_prev = rospy.get_time()
        self.err_prev = 0.0
        self.err_sum = 0.0
        ######### Your code ends here #########

    def control(self, err, t):
        # computer PID control action here
        ######### Your code starts here #########
        dt = t - self.t_prev
        if dt <= 0.0:
            return 0.0  # No control action if time has not advanced
        self.t_prev = t

        de = err - self.err_prev
        self.err_prev = err
        self.err_sum += err * dt

        u = self.kP * err + self.kI * self.err_sum + self.kD * de / dt + self.kS * (1 if err > 0 else -1 if err < 0 else 0)
        return max(self.u_min, min(self.u_max, u))
  
######### Your code ends here #########


class Particle:
    def __init__(self, x: float, y: float, theta: float, log_p: float):
        self.x = x
        self.y = y
        self.theta = theta
        self.log_p = log_p

    def __str__(self) -> str:
        return f"Particle<pose: {self.x, self.y, self.theta}, log_p: {self.log_p}>"


class ParticleFilter:

    def __init__(
        self,
        map_: Map,
        n_particles: int,
        translation_variance: float,
        rotation_variance: float,
        measurement_variance: float,
    ):
        self.particles_visualization_pub = rospy.Publisher("/pf_particles", PoseArray, queue_size=10)
        self.estimate_visualization_pub = rospy.Publisher("/pf_estimate", PoseStamped, queue_size=10)

        # Initialize uniformly-distributed particles
        ######### Your code starts here #########
        self._map = map_
        self._n_particles = n_particles
        self._translation_variance = translation_variance
        self._rotation_variance = rotation_variance
        self._measurement_variance = measurement_variance

        # Get map boundaries from the AABB (Axis-Aligned Bounding Box)
        x_min, y_min = self._map.bottom_left
        x_max, y_max = self._map.top_right

        self._particles = []
        for _ in range(self._n_particles):
            # Sample random position within map limits
            rx = uniform(x_min, x_max)
            ry = uniform(y_min, y_max)
            # Sample random orientation between -pi and pi
            rtheta = uniform(-math.pi, math.pi)
            
            # Initial log-probability is 0 (equal probability)
            self._particles.append(Particle(rx, ry, rtheta, 0.0))
        
     
        ######### Your code ends here #########

    def visualize_particles(self):
        pa = PoseArray()
        pa.header.frame_id = "odom"
        pa.header.stamp = rospy.Time.now()
        for particle in self._particles:
            pose = Pose()
            pose.position = Point(particle.x, particle.y, 0.01)
            q_np = quaternion_from_euler(0, 0, float(particle.theta))
            pose.orientation = Quaternion(*q_np.tolist())
            pa.poses.append(pose)
        self.particles_visualization_pub.publish(pa)

    def visualize_estimate(self):
        ps = PoseStamped()
        ps.header.frame_id = "odom"
        ps.header.stamp = rospy.Time.now()
        x, y, theta = self.get_estimate()
        pose = Pose()
        pose.position = Point(x, y, 0.01)
        q_np = quaternion_from_euler(0, 0, float(theta))
        pose.orientation = Quaternion(*q_np.tolist())
        ps.pose = pose
        self.estimate_visualization_pub.publish(ps)

    def move_by(self, delta_x, delta_y, delta_theta):
        delta_theta = angle_to_neg_pi_to_pi(delta_theta)

        # Propagate motion of each particle
        ######### Your code starts here #########
        # Commanded distance from the deltas
        d = math.sqrt(delta_x**2 + delta_y**2)
        
        # σd = 0.01, σΘ = 0.05
        for p in self._particles:
            # Add noise to the commanded motion
         
            noisy_theta_increment = delta_theta + np.random.normal(0, self._rotation_variance)
            noisy_dist = d + np.random.normal(0, self._translation_variance)
            
            # Update particle pose
            # Update heading first
            p.theta = angle_to_neg_pi_to_pi(p.theta + noisy_theta_increment)
            # Move in the direction of the new heading
            p.x += noisy_dist * math.cos(p.theta)
            p.y += noisy_dist * math.sin(p.theta)
        

        ######### Your code ends here #########

    def measure(self, z: float, scan_angle_in_rad: float):
        """Update the particles based on the measurement `z` at the given `scan_angle_in_rad`.

        Args:
            z: distance to an obstacle
            scan_angle_in_rad: Angle in the robots frame where the scan was taken
        """

        # Calculate posterior probabilities and resample
        ######### Your code starts here #########
        log_weights = []
        
        for p in self._particles:
            # Get expected distance from map for this particle's pose
        
            expected_z = self._map.closest_distance((p.x, p.y), p.theta + scan_angle_in_rad)
            
            if expected_z is None:
                # Particle is outside map or ray doesn't hit anything
                p.log_p = -inf
            else:
                # Update log-probability: log(P_new) = log(P_old) + log(P_sensor)
                # σs = 0.1 (measurement_variance)
                log_p_sensor = scipy.stats.norm(loc=expected_z, scale=self._measurement_variance).logpdf(z)
                p.log_p += log_p_sensor
            
            log_weights.append(p.log_p)

        # Resample using weights
        # Convert log-probs back to linear weights for the choices() function
        # Subtract max(log_p) before exp() for numerical stability (Softmax trick)
        max_log = max(log_weights)
        weights = [math.exp(lw - max_log) if lw != -inf else 0 for lw in log_weights]
        sum_weights = sum(weights)
        
        if sum_weights > 0:
            normalized_weights = [w / sum_weights for w in weights]
            # Use np.random.choice to pick new particles based on weights
            new_indices = np.random.choice(len(self._particles), size=self._n_particles, p=normalized_weights)
            
            new_particles = []
            for idx in new_indices:
                template = self._particles[idx]
                # Reset log_p to 0 (equal probability) after resampling
                new_particles.append(Particle(template.x, template.y, template.theta, 0.0))
            self._particles = new_particles

        ######### Your code ends here #########

    def get_estimate(self) -> Tuple[float, float, float]:
        # Estimate robot's location using particle weights
        ######### Your code starts here #########
        if not self._particles:
            return 0.0, 0.0, 0.0
            
        avg_x = sum(p.x for p in self._particles) / self._n_particles
        avg_y = sum(p.y for p in self._particles) / self._n_particles
        
        # For angles, we must average the vectors to avoid the 0/2pi wrap-around issue
        avg_cos = sum(math.cos(p.theta) for p in self._particles) / self._n_particles
        avg_sin = sum(math.sin(p.theta) for p in self._particles) / self._n_particles
        avg_theta = math.atan2(avg_sin, avg_cos)
        
        return avg_x, avg_y, avg_theta

        ######### Your code ends here #########


class Controller:
    def __init__(self, particle_filter: ParticleFilter):
        rospy.init_node("particle_filter_controller", anonymous=True)
        self._particle_filter = particle_filter
        self._particle_filter.visualize_particles()

        #
        self.current_position = None
        self.laserscan = None
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        self.laserscan_sub = rospy.Subscriber("/scan", LaserScan, self.robot_laserscan_callback)
        self.robot_ctrl_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.pointcloud_pub = rospy.Publisher("/scan_pointcloud", PointCloud, queue_size=10)
        self.target_position_pub = rospy.Publisher("/waypoints", MarkerArray, queue_size=10)

        while ((self.current_position is None) or (self.laserscan is None)) and (not rospy.is_shutdown()):
            rospy.loginfo("waiting for odom and laserscan")
            rospy.sleep(0.1)

    def odom_callback(self, msg):
        pose = msg.pose.pose
        orientation = pose.orientation
        _, _, theta = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.current_position = {"x": pose.position.x, "y": pose.position.y, "theta": theta}

    def robot_laserscan_callback(self, msg: LaserScan):
        self.laserscan = msg

    def visualize_laserscan_ranges(self, idx_groups: List[Tuple[int, int]]):
        """Helper function to visualize ranges of sensor readings from the laserscan lidar.

        Example usage for visualizing the first 10 and last 10 degrees of the laserscan:
            `self.visualize_laserscan_ranges([(0, 10), (350, 360)])`
        """
        pcd = PointCloud()
        pcd.header.frame_id = "odom"
        pcd.header.stamp = rospy.Time.now()
        for idx_low, idx_high in idx_groups:
            for idx, d in enumerate(self.laserscan.ranges[idx_low:idx_high]):
                if d == inf:
                    continue
                angle = math.radians(idx) + self.current_position["theta"]
                x = d * math.cos(angle) + self.current_position["x"]
                y = d * math.sin(angle) + self.current_position["y"]
                z = 0.1
                pcd.points.append(Point32(x=x, y=y, z=z))
                pcd.channels.append(ChannelFloat32(name="rgb", values=(0.0, 1.0, 0.0)))
        self.pointcloud_pub.publish(pcd)

    def visualize_position(self, x: float, y: float):
        marker_array = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "waypoints"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position = Point(x, y, 0.0)
        marker.pose.orientation = Quaternion(0, 0, 0, 1)
        marker.scale = Vector3(0.075, 0.075, 0.1)
        marker.color = ColorRGBA(0.0, 0.0, 1.0, 0.5)
        marker_array.markers.append(marker)
        self.target_position_pub.publish(marker_array)

    def take_measurements(self):
        # Take measurement using LIDAR
        ######### Your code starts here #########
        # NOTE: with more than 2 angles the particle filter will converge too quickly, so with high likelihood the
        # correct neighborhood won't be found.
        if self.laserscan is None:
            return
            
        sample_indices = [0, 90]
        
        for idx in sample_indices:
            z = self.laserscan.ranges[idx]
            
          
            if math.isinf(z) or z <= 0.1:
                continue
                
            
            scan_angle_rad = math.radians(idx)
        
            self._particle_filter.measure(z, scan_angle_rad)
        
       
        self._particle_filter.visualize_particles()
        self._particle_filter.visualize_estimate()
        

        ######### Your code ends here #########

    def autonomous_exploration(self):
        """Randomly explore the environment here, while making sure to call `take_measurements()` and
        `_particle_filter.move_by()`. The particle filter should converge on the robots position eventually.

        Note that the following visualizations functions are available:
            visualize_position(...)
            visualize_laserscan_ranges(...)
        """
        # Robot autonomously explores environment while it localizes itself
        ######### Your code starts here #########
        rospy.loginfo("Autonomous Localization...")
        steps = 0 
        
        while not rospy.is_shutdown():
            # Update Particle Filter with current LIDAR data
            self.take_measurements()
            
            # Pause the automatic exploration
            estimate_x, estimate_y, estimate_theta = self._particle_filter.get_estimate()
            
            # calculate spread 
            particles = self._particle_filter._particles
            mean_x = sum(p.x for p in particles) / len(particles)
            mean_y = sum(p.y for p in particles) / len(particles)
            spread = math.sqrt(sum((p.x-mean_x)**2 + (p.y-mean_y)**2 for p in particles) / len(particles))
            
            
            rospy.loginfo(f"Steps: {steps}, Current Spread: {spread:.4f} m")
            
            
            if steps >= 5 and spread < 0.05:
                rospy.loginfo("Achieved accuracy.")
                rospy.loginfo(f"Final Estimate: x={estimate_x:.2f}, y={estimate_y:.2f}, theta={estimate_theta:.2f}")
                self.robot_ctrl_pub.publish(Twist()) 
                self._particle_filter.visualize_estimate()
                break
    
            # Check LIDAR ranges 
            # use a threshold (0.5m) to avoid hitting walls
            front_clearance = self.laserscan.ranges[0]
            
            if front_clearance > 0.5 and not math.isinf(front_clearance):
                # Path is clear: Move forward
                step_size = 0.3
                self.forward_action(step_size)
                self._particle_filter.move_by(step_size, 0, 0)
            else:
                # if a wall is detected rotate 90 degrees (pi/2)
                turn_amount = math.pi / 2
                target_theta = angle_to_neg_pi_to_pi(self.current_position["theta"] + turn_amount)
                self.rotate_action(target_theta)

                self._particle_filter.move_by(0, 0, turn_amount)
                
            steps += 1 
            rospy.sleep(0.1)



        ######### Your code ends here #########

    def forward_action(self, distance: float):
        # Robot moves forward by a set amount during manual control
        ######### Your code starts here #########
        rate = rospy.Rate(10)
        # TurtleBot3 Max Linear: ~0.22 m/s. Use a safe max of 0.15.
        linear_pid = PIDController(kP=1.2, kI=0.01, kD=0.05, kS=0.0, u_min=-0.15, u_max=0.15)
        
        start_x = self.current_position["x"]
        start_y = self.current_position["y"]
        
        last_time = rospy.get_time()
        
        while not rospy.is_shutdown():
            # Current distance from start
            curr_x = self.current_position["x"]
            curr_y = self.current_position["y"]
            moved = math.sqrt((curr_x - start_x)**2 + (curr_y - start_y)**2)
            
            error = abs(distance) - moved
            
            if error < 0.02: # Stop within 2cm
                break
                
            current_time = rospy.get_time()
            
            output = linear_pid.control(error, current_time)
            cmd = Twist()
            # In case moving backward
            direction = 1 if distance > 0 else -1
            cmd.linear.x = direction * abs(output)
            self.robot_ctrl_pub.publish(cmd)
            rate.sleep()
        self.robot_ctrl_pub.publish(Twist()) # Force stop
        ######### Your code ends here #########

    def rotate_action(self, goal_theta: float):
        # Robot turns by a set amount during manual control
        ######### Your code starts here #########
        rate = rospy.Rate(10)
        # TurtleBot3 Max Angular: ~2.84 rad/s.
        angular_pid = PIDController(kP=2.0, kI=0.1, kD=0.05, kS=0.0, u_min=-1.0, u_max=1.0)
        
        while not rospy.is_shutdown():
            error = angle_to_neg_pi_to_pi(goal_theta - self.current_position["theta"])
            
            if abs(error) < 0.05: # Stop within ~3 degrees
                break
                
            cmd = Twist()
            cmd.angular.z = angular_pid.control(error, rospy.get_time())
            self.robot_ctrl_pub.publish(cmd)
            rate.sleep()
            
        self.robot_ctrl_pub.publish(Twist()) # Force stop

        ######### Your code ends here #########


""" Example usage

rosrun development lab8_9.py --map_filepath src/csci455l/scripts/lab8_9_map.json
"""


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--map_filepath", type=str, required=True)
    args = parser.parse_args()
    with open(args.map_filepath, "r") as f:
        map_ = json.load(f)
        obstacles = map_["obstacles"]
        map_aabb = map_["map_aabb"]

    map_ = Map(obstacles, map_aabb)
    num_particles = 200
    translation_variance = 0.25
    rotation_variance = 0.15
    measurement_variance = 0.1
    particle_filter = ParticleFilter(map_, num_particles, translation_variance, rotation_variance, measurement_variance)
    controller = Controller(particle_filter)

    try:
        # Manual control
        goal_theta = 0
        controller.take_measurements()
        print("Select Mode: [m] Manual, [a] Autonomous")
        mode_input = input(">> ").lower()
        while not rospy.is_shutdown():
          
            if mode_input == 'm':
               print("\nEnter 'a', 'w', 's', 'd' to move the robot:")
            
               uinput = input("")
               if uinput == "w": # forward
                   ######### Your code starts here #########
                   # Forward 0.25 m
                   dist = 0.25 
                   controller.forward_action(dist)
                   particle_filter.move_by(dist, 0, 0)
                   print("distance forward 0.25 m")
                ######### Your code ends here #########
               elif uinput == "a": # left
                   ######### Your code starts here #########
                   # Turn left pi/2
                   target_angle = angle_to_neg_pi_to_pi(controller.current_position["theta"]+ math.pi/2)
                   controller.rotate_action(target_angle)
                   particle_filter.move_by(0, 0, math.pi/2)
                   print("angle turned left 90 degrees")
                   ######### Your code ends here #########
               elif uinput == "d": #right
                   ######### Your code starts here #########
                   # Turn right pi/2 
                   target_angle = angle_to_neg_pi_to_pi(controller.current_position["theta"] - math.pi/2)
                   controller.rotate_action(target_angle)
                   particle_filter.move_by(0, 0, -math.pi/2)
                   print("angle turned right 90 degrees")
                ######### Your code ends here #########
               elif uinput == "s": # backwards
                ######### Your code starts here ########
                   dist = -0.25 
                   controller.forward_action(dist)
                   particle_filter.move_by(dist, 0, 0)
                   print("distance backward 0.25m")
                ######### Your code ends here #########
               else:
                   print("Invalid input")
            ######### Your code starts here #########
            controller.take_measurements()
            ######### Your code ends here #########

        # Autonomous exploration
        ######### Your code starts here #########
            if mode_input == 'a':
                print("Autonomous exploration...")
                controller.autonomous_exploration()
                
                break
        ######### Your code ends here #########

    except rospy.ROSInterruptException:
        print("Shutting down...")
