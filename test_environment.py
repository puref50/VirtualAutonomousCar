import random, time, math, cv2
import numpy as np
import gymnasium
from gymnasium import spaces
import carla
from tensorflow.keras.models import load_model

# Key parameters
SECONDS_PER_EPISODE = 150
N_CHANNELS, HEIGHT, WIDTH = 3, 240, 320
SPIN = 10
HEIGHT_REQUIRED_PORTION = 0.5
WIDTH_REQUIRED_PORTION = 0.9
SHOW_PREVIEW = True
SEED = 123

class CarEnv(gymnasium.Env):
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width, im_height = WIDTH, HEIGHT
    CAMERA_POS_Z, CAMERA_POS_X = 1.3, 1.4
    PREFERRED_SPEED, SPEED_THRESHOLD = 30, 2

    def __init__(self):
        super(CarEnv, self).__init__()
        self.action_space = spaces.MultiDiscrete([9])
        
        self.height_from = int(HEIGHT * (1 - HEIGHT_REQUIRED_PORTION))
        self.width_from = int((WIDTH - WIDTH * WIDTH_REQUIRED_PORTION) / 2)
        self.width_to = self.width_from + int(WIDTH_REQUIRED_PORTION * WIDTH)
        self.new_height = HEIGHT - self.height_from
        self.new_width = self.width_to - self.width_from
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(7, 18, 8), dtype=np.float32)

        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(4.0)
        self.client.load_world('Town10HD')
        self.world = self.client.get_world()
        self.settings = self.world.get_settings()
        self.settings.no_rendering_mode = not self.SHOW_CAM
        self.world.apply_settings(self.settings)

        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.cnn_model = load_model("CNN_image_model.h5", compile=False)
        self.cnn_model.compile()

        if self.SHOW_CAM:
            self.spectator = self.world.get_spectator()

    def cleanup(self):
        for sensor in self.world.get_actors().filter('*sensor*'):
            sensor.destroy()
        for actor in self.world.get_actors().filter('*vehicle*'):
            actor.destroy()
        cv2.destroyAllWindows()

    def maintain_speed(self, s):
        if s >= self.PREFERRED_SPEED:
            return 0
        elif s < self.PREFERRED_SPEED - self.SPEED_THRESHOLD:
            return 0.7
        else:
            return 0.3

    def apply_cnn(self, im):
        img = np.float32(im) / 255
        img = np.expand_dims(img, axis=0)
        output = self.cnn_model([img, 0], training=False)
        return np.squeeze(output)

    def step(self, action):
        trans = self.vehicle.get_transform()
        if self.SHOW_CAM:
            self.spectator.set_transform(
                carla.Transform(trans.location + carla.Location(z=20), carla.Rotation(yaw=-180, pitch=-90))
            )

        self.step_counter += 1
        steer = [-0.9, -0.25, -0.1, -0.05, 0.0, 0.05, 0.1, 0.25, 0.9][action[0]]
        if self.step_counter % 50 == 0:
            print('steer input:', steer)

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))
        throttle = self.maintain_speed(kmh)
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer))

        distance_travelled = self.initial_location.distance(self.vehicle.get_location())
        cam = self.front_camera
        if self.SHOW_CAM:
            cv2.imshow('Sem Camera', cam)
            cv2.waitKey(1)

        # Check steering lock
        lock_duration = 0
        if not self.steering_lock:
            if abs(steer) > 0.6:
                self.steering_lock = True
                self.steering_lock_start = time.time()
        else:
            if abs(steer) > 0.6:
                lock_duration = time.time() - self.steering_lock_start

        # Reward function for collision, lane invasion and steering lock
        reward, done = 0, False
        if self.collision_hist:
            done, reward = True, -300
        elif self.lane_invade_hist:
            done, reward = False, -300
        elif lock_duration > 3:
            done, reward = False, -150
        elif lock_duration > 1:
            reward -= 20

        # Reward function for lane keeping
        waypoint = self.world.get_map().get_waypoint(self.vehicle.get_location(), project_to_road=True)
        if waypoint and not waypoint.is_intersection:
            lane_center = waypoint.transform.location
            distance = self.vehicle.get_location().distance(lane_center)
            yaw_diff = abs(self.vehicle.get_transform().rotation.yaw - waypoint.transform.rotation.yaw) % 360
            yaw_diff = min(yaw_diff, 360 - yaw_diff)
            reward -= min(distance, 2.0)
            if yaw_diff > 10:
                reward -= (yaw_diff / 180.0) * 2

        # Reward function for surviving
        if distance_travelled < 30:
            reward -= 1
        elif distance_travelled < 50:
            reward += 1
        else:
            reward += 2

        reward += 0.1

        self.episode_total_reward += reward
        cur_dist = self.initial_location.distance(self.vehicle.get_location())
        self.episode_total_distance = cur_dist
        self.episode_lane_invasion_count += len(self.lane_invade_hist)
        self.lane_invade_hist = [] 

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True
            self.cleanup()

        self.image_for_CNN = self.apply_cnn(cam[self.height_from:, self.width_from:self.width_to])

        info = {
			"episode_reward": self.episode_total_reward,
			"episode_distance": self.episode_total_distance,
			"lane_invasions": self.episode_lane_invasion_count
		}
        return self.image_for_CNN, reward, done, done, info

    def reset(self, seed=SEED):
        # Delete people and other vehicles
        for actor in self.world.get_actors().filter('*walker*'):
            actor.destroy()
        for actor in self.world.get_actors().filter('*vehicle*'):
            actor.destroy()
        
        self.collision_hist, self.lane_invade_hist, self.actor_list = [], [], []
        self.transform = random.choice(self.world.get_map().get_spawn_points())

        self.episode_total_reward = 0
        self.episode_total_distance = 0
        self.episode_lane_invasion_count = 0

        self.vehicle = None
        while self.vehicle is None:
            try:
                self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
            except:
                pass
        self.actor_list.append(self.vehicle)
        self.initial_location = self.vehicle.get_location()

        # Add semantic camera
        cam_bp = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        cam_bp.set_attribute("image_size_x", f"{self.im_width}")
        cam_bp.set_attribute("image_size_y", f"{self.im_height}")
        cam_bp.set_attribute("fov", "90")
        cam_trans = carla.Transform(carla.Location(z=self.CAMERA_POS_Z, x=self.CAMERA_POS_X))
        self.sensor = self.world.spawn_actor(cam_bp, cam_trans, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(2)

        # Give a random initial yaw angle
        angle_adj = random.randrange(-SPIN, SPIN, 1)
        trans = self.vehicle.get_transform()
        trans.rotation.yaw += angle_adj
        self.vehicle.set_transform(trans)

        # Add sensor for collision and lane invasion
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, cam_trans, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda e: self.collision_data(e))

        lanesensor = self.blueprint_library.find("sensor.other.lane_invasion")
        self.lanesensor = self.world.spawn_actor(lanesensor, cam_trans, attach_to=self.vehicle)
        self.actor_list.append(self.lanesensor)
        self.lanesensor.listen(lambda e: self.lane_data(e))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.steering_lock = False
        self.steering_lock_start = None
        self.step_counter = 0
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        self.image_for_CNN = self.apply_cnn(self.front_camera[self.height_from:, self.width_from:self.width_to])
        return self.image_for_CNN, {}

    def process_img(self, image):
        image.convert(carla.ColorConverter.CityScapesPalette)
        arr = np.array(image.raw_data).reshape((self.im_height, self.im_width, 4))[:, :, :3]
        self.front_camera = arr

    def collision_data(self, event):
        self.collision_hist.append(event)

    def lane_data(self, event):
        self.lane_invade_hist.append(event)
