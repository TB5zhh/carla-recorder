#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""
Script that render multiple sensors in the same pygame window

By default, it renders four cameras, one LiDAR and one Semantic LiDAR.
It can easily be configure for any different number of sensors. 
To do that, check lines 290-308.
"""

import glob
from re import S
import cv2
import sys
from IPython import embed
from tqdm import tqdm
import os

chk = True
FN = "6view_with_anomaly"


def build_folder(foldername):
    if os.path.exists(foldername):
        print("Folder exists!")
    else:
        os.makedirs(foldername)


def path_generator(fn):
    os.makedirs(fn, exist_ok=True)
    fid = str(len(os.listdir(fn)) + 1)
    os.makedirs(os.path.join(fn, fid), exist_ok=True)
    os.makedirs(os.path.join(fn, fid, 'mask_x'), exist_ok=True)
    os.makedirs(os.path.join(fn, fid, 'mask_v'), exist_ok=True)
    os.makedirs(os.path.join(fn, fid, 'rgb_x'), exist_ok=True)
    os.makedirs(os.path.join(fn, fid, 'rgb_v'), exist_ok=True)
    os.makedirs(os.path.join(fn, fid, 'depth_x'), exist_ok=True)
    os.makedirs(os.path.join(fn, fid, 'depth_v'), exist_ok=True)
    return fid


try:
    sys.path.append(
        glob.glob('../carla/dist/carla-*%d.%d-%s.egg' %
                  (sys.version_info.major, sys.version_info.minor, 'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import argparse
import random
import time
import logging
import numpy as np

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

angle_filter = lambda angle: (angle + 45) % 90 < 50 and (angle + 45) % 90 > 40


class CustomTimer:

    def __init__(self):
        try:
            self.timer = time.perf_counter
        except AttributeError:
            self.timer = time.time

    def time(self):
        return self.timer()


class DisplayManager:

    def __init__(self, grid_size, window_size):
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(window_size, pygame.HWSURFACE | pygame.DOUBLEBUF)

        self.grid_size = grid_size
        self.window_size = window_size
        self.sensor_list = []

    def get_window_size(self):
        return [int(self.window_size[0]), int(self.window_size[1])]

    def get_display_size(self):
        return [int(self.window_size[0] / self.grid_size[1]), int(self.window_size[1] / self.grid_size[0])]

    def get_display_offset(self, gridPos):
        dis_size = self.get_display_size()
        return [int(gridPos[1] * dis_size[0]), int(gridPos[0] * dis_size[1])]

    def add_sensor(self, sensor):
        self.sensor_list.append(sensor)

    def get_sensor_list(self):
        return self.sensor_list

    def render(self):
        if not self.render_enabled():
            return

        for s in self.sensor_list:
            s.render()

        pygame.display.flip()

    def destroy(self):
        for s in self.sensor_list:
            s.destroy()

    def render_enabled(self):
        return self.display != None


class SensorManager:

    def __init__(self, world, display_man, sensor_type, transform, attached, sensor_options, display_pos, file_dir, fid):
        self.surface = None
        self.world = world
        self.display_man = display_man
        self.display_pos = display_pos
        self.fn = file_dir
        self.fid = fid
        self.sensor = self.init_sensor(sensor_type, transform, attached, sensor_options)
        self.sensor_options = sensor_options
        self.timer = CustomTimer()

        self.time_processing = 0.0
        self.tics_processing = 0


        self.display_man.add_sensor(self)

    def init_sensor(self, sensor_type, transform, attached, sensor_options):
        if (sensor_type == 'xRGBCamera') or (sensor_type == 'vRGBCamera'):
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            disp_size = self.display_man.get_display_size()
            camera_bp.set_attribute('image_size_x', str(disp_size[0]))
            camera_bp.set_attribute('image_size_y', str(disp_size[1]))

            for key in sensor_options:
                camera_bp.set_attribute(key, sensor_options[key])

            camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)
            if sensor_type == 'xRGBCamera':
                if chk:
                    with open(f"{self.fn}/{self.fid}/path.txt", 'a+') as f:
                        f.write("Infra Camera: " + str(transform) + "\n")
                camera.listen(self.save_rgb_image_x)
            else:
                camera.listen(self.save_rgb_image_v)

            return camera

        elif (sensor_type == "xSemanticCamera") or (sensor_type == "vSemanticCamera"):
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
            disp_size = self.display_man.get_display_size()
            camera_bp.set_attribute('image_size_x', str(disp_size[0]))
            camera_bp.set_attribute('image_size_y', str(disp_size[1]))

            for key in sensor_options:
                camera_bp.set_attribute(key, sensor_options[key])

            camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)
            if sensor_type == "xSemanticCamera":
                camera.listen(self.save_semantic_image_x)
            else:
                camera.listen(self.save_semantic_image_v)

            return camera

        elif (sensor_type == "xDepthCamera") or (sensor_type == "vDepthCamera"):
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.depth')
            disp_size = self.display_man.get_display_size()
            camera_bp.set_attribute('image_size_x', str(disp_size[0]))
            camera_bp.set_attribute('image_size_y', str(disp_size[1]))

            for key in sensor_options:
                camera_bp.set_attribute(key, sensor_options[key])

            camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)
            if sensor_type == "xDepthCamera":
                camera.listen(self.save_depth_image_x)
            else:
                camera.listen(self.save_depth_image_v)

            return camera

        else:
            return None

    def get_sensor(self):
        return self.sensor

    ############################################## save data ###################################################
    def save_rgb_image(self, image):
        t_start = self.timer.time()

        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        t_end = self.timer.time()
        self.time_processing += (t_end - t_start)
        self.tics_processing += 1
        return array

    def save_rgb_image_x(self, image):
        array = self.save_rgb_image(image)
        array = array[:, :, ::-1]
        if chk:
            cv2.imwrite(self.fn + "/{}/rgb_x/{}.png".format(self.fid, str(self.tics_processing)), array)

    def save_rgb_image_v(self, image):
        array = self.save_rgb_image(image)
        array = array[:, :, ::-1]
        if chk:
            cv2.imwrite(self.fn + "/{}/rgb_v/{}.png".format(self.fid, str(self.tics_processing)), array)
            with open(f'{self.fn}/{self.fid}/path.txt', 'a+') as f:
                f.write(str(self.tics_processing) + ": " + str(self.sensor.get_transform()) + "\n")

    def save_semantic_image(self, image):
        t_start = self.timer.time()

        image.convert(carla.ColorConverter.CityScapesPalette)
        # image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        t_end = self.timer.time()
        self.time_processing += (t_end - t_start)
        self.tics_processing += 1
        return array

    def save_semantic_image_x(self, image):
        array = self.save_semantic_image(image)
        array = array[:, :, ::-1]
        if chk:
            cv2.imwrite(self.fn + "/{}/mask_x/{}.png".format(self.fid, str(self.tics_processing)), array)

    def save_semantic_image_v(self, image):
        array = self.save_semantic_image(image)
        array = array[:, :, ::-1]
        if chk:
            cv2.imwrite(self.fn + "/{}/mask_v/{}.png".format(self.fid, str(self.tics_processing)), array)

    def save_depth_image(self, image):
        t_start = self.timer.time()

        # image.convert(carla.ColorConverter.LogarithmicDepth)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        t_end = self.timer.time()
        self.time_processing += (t_end - t_start)
        self.tics_processing += 1
        return array

    def save_depth_image_x(self, image):
        array = self.save_depth_image(image)
        array = array[:, :, ::-1]
        if chk:
            cv2.imwrite(self.fn + "/{}/depth_x/{}.png".format(self.fid, str(self.tics_processing)), array)

    def save_depth_image_v(self, image):
        array = self.save_depth_image(image)
        array = array[:, :, ::-1]
        if chk:
            cv2.imwrite(self.fn + "/{}/depth_v/{}.png".format(self.fid, str(self.tics_processing)), array)

    def render(self):
        if self.surface is not None:
            offset = self.display_man.get_display_offset(self.display_pos)
            self.display_man.display.blit(self.surface, offset)

    def destroy(self):
        self.sensor.destroy()


def run_simulation(args, client):
    """This function performed one test run using the args parameters
    and connecting to the carla client passed.
    """

    display_manager = None
    vehicle = None
    timer = CustomTimer()

    vehicles_list = []
    walkers_list = []
    all_id = []
    ob_list = []
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    synchronous_master = False
    random.seed(args.seed if args.seed is not None else int(time.time()))

    world = client.get_world()
    original_settings = world.get_settings()
    try:

        # Getting the world and
        print(client.get_available_maps())

        # weather manager
        def static_weather(id):
            if id == 1:
                return carla.WeatherParameters(cloudiness=0.0, precipitation=0.0, fog_density=0.0, sun_altitude_angle=70.0)
            elif id == 2:
                return carla.WeatherParameters(cloudiness=0.0, precipitation=0.0, fog_density=40.0, sun_altitude_angle=70.0)
            elif id == 3:
                return carla.WeatherParameters(cloudiness=50.0, precipitation=80.0, fog_density=30.0, sun_altitude_angle=60.0)
            else:
                return carla.WeatherParameters(cloudiness=0.0, precipitation=0.0, fog_density=0.0, sun_altitude_angle=0.0)

        print("weather id:", args.weather)
        weather = static_weather(args.weather)
        world.set_weather(weather)

        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        if args.respawn:
            traffic_manager.set_respawn_dormant_vehicles(True)
        if args.hybrid:
            traffic_manager.set_hybrid_physics_mode(True)
            traffic_manager.set_hybrid_physics_radius(70.0)
        if args.seed is not None:
            traffic_manager.set_random_device_seed(args.seed)

        settings = world.get_settings()
        if args.sync:
            # traffic_manager = client.get_trafficmanager(8000)
            traffic_manager.set_synchronous_mode(True)
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)

        # Instanciating the vehicle to which we attached the sensors
        bp = world.get_blueprint_library().filter('charger_2020')[0]
        while True:
            spawn_point_id = random.randint(0, len(world.get_map().get_spawn_points()) - 1)
            spawn_point = world.get_map().get_spawn_points()[spawn_point_id]
            if angle_filter(spawn_point.rotation.yaw):
                break
        spawn_point_id = 141
        spawn_point = world.get_map().get_spawn_points()[spawn_point_id]
        # id: 141 spawn_point: Transform(Location(x=65.235275, y=13.414804, z=0.600000), Rotation(pitch=0.000000, yaw=-179.840790, roll=0.000000))
        spawn_point = carla.Transform(
                carla.Location(x=70.235275, y=13.414804, z=0.600000),
                carla.Rotation(pitch=0.000000, yaw=-179.840790, roll=0.000000),
            )
        print("id:", spawn_point_id, "spawn_point:", spawn_point)
        vehicle = world.spawn_actor(bp, spawn_point)
        vehicles_list.append(vehicle)
        vehicle.set_autopilot(True)

        # set location and posture of infra sensors
        anomaly_distance = random.randint(50, 50)
        camera_distance = random.randint(anomaly_distance + 6, anomaly_distance + 6)
        camera_yaw = round(spawn_point.rotation.yaw + 180) % 360
        camera_pitch = -random.randint(15, 15)

        calib = 0
        if spawn_point.rotation.yaw < 45 and spawn_point.rotation.yaw > -45:
            infra_sensor_pos = carla.Transform(
                # carla.Location(x=spawn_point.location.x + camera_distance, y=spawn_point.location.y, z=4),
                carla.Location(x=spawn_point.location.x + camera_distance, y=spawn_point.location.y, z=4),
                carla.Rotation(pitch=camera_pitch, yaw=camera_yaw, roll=0.000000),
            )
        elif spawn_point.rotation.yaw > 45 and spawn_point.rotation.yaw < 135:
            infra_sensor_pos = carla.Transform(
                carla.Location(x=spawn_point.location.x, y=spawn_point.location.y + camera_distance, z=4),
                carla.Rotation(pitch=camera_pitch, yaw=camera_yaw, roll=0.000000),
            )
        elif spawn_point.rotation.yaw > 135 or spawn_point.rotation.yaw < -135:
            infra_sensor_pos = carla.Transform(
                carla.Location(x=spawn_point.location.x - camera_distance, y=spawn_point.location.y, z=4),
                carla.Rotation(pitch=camera_pitch, yaw=camera_yaw, roll=0.000000),
            )
        elif spawn_point.rotation.yaw < -45 and spawn_point.rotation.yaw > -135:
            infra_sensor_pos = carla.Transform(
                carla.Location(x=spawn_point.location.x, y=spawn_point.location.y - camera_distance, z=4),
                carla.Rotation(pitch=camera_pitch, yaw=camera_yaw, roll=0.000000),
            )

        infra_sensor_pos = carla.Transform(
            carla.Location(x=-35, y=16.414804, z=4),
            carla.Rotation(pitch=camera_pitch, yaw=0, roll=0.000000),
        )
        # infra_sensor_pos = carla.Transform(carla.Location(x=-45, y=90, z=5), carla.Rotation(pitch=0.000000, yaw=-90, roll=0.000000))

        # Display Manager organize all the sensors an its display in a window
        # If can easily configure the grid and the total window size
        grid_size = [3, 2]
        display_manager = DisplayManager(grid_size=grid_size, window_size=[args.width * grid_size[1], args.height * grid_size[0]])

        # Then, SensorManager can be used to spawn RGBCamera, LiDARs and SemanticLiDARs as needed
        # and assign each of them to a grid position,
        fid = path_generator(args.file_dir)
        SensorManager(world,
                      display_manager,
                      'vRGBCamera',
                      carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=+00)),
                      vehicle, {},
                      display_pos=[0, 0],
                      file_dir=args.file_dir,
                      fid=fid)
        SensorManager(world,
                      display_manager,
                      'vSemanticCamera',
                      carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=+00)),
                      vehicle, {},
                      display_pos=[1, 0],
                      file_dir=args.file_dir,
                      fid=fid)
        SensorManager(world,
                      display_manager,
                      'vDepthCamera',
                      carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=+00)),
                      vehicle, {},
                      display_pos=[2, 0],
                      file_dir=args.file_dir,
                      fid=fid)
        SensorManager(world, display_manager, 'xRGBCamera', infra_sensor_pos, None, {}, display_pos=[0, 1], file_dir=args.file_dir, fid=fid)
        SensorManager(world, display_manager, 'xSemanticCamera', infra_sensor_pos, None, {}, display_pos=[1, 1], file_dir=args.file_dir, fid=fid)
        SensorManager(world, display_manager, 'xDepthCamera', infra_sensor_pos, None, {}, display_pos=[2, 1], file_dir=args.file_dir, fid=fid)
        #######################################################################################################################################################
        # # generate obstacle

        # embed()
        anomaly_object_type = random.randint(0, 5)
        anomaly_object_type = 6
        # yaw = random.random() * 180 - 90
        # yaw = 0
        yaw = 45
        if anomaly_object_type == 0:  # container
            parameters = {'z': 0., 'pitch': 0., 'yaw': yaw, 'roll': 0.}
            ob_bp = world.get_blueprint_library().find('static.prop.container')
        elif anomaly_object_type == 1:  # Motor helmet
            parameters = {'z': 0.25, 'pitch': 0., 'yaw': yaw, 'roll': 0.}
            ob_bp = world.get_blueprint_library().find('static.prop.motorhelmet')
        elif anomaly_object_type == 2:  # Guitar case
            parameters = {'z': 0.07, 'pitch': 0., 'yaw': yaw, 'roll': 90.}
            ob_bp = world.get_blueprint_library().find('static.prop.guitarcase')
        elif anomaly_object_type == 3:  # Shopping bag
            parameters = {'z': 0.04, 'pitch': 0., 'yaw': yaw, 'roll': 99.}
            ob_bp = world.get_blueprint_library().find('static.prop.shoppingbag')
        elif anomaly_object_type == 4:  # Shopping cart
            parameters = {'z': 1.08, 'pitch': 0., 'yaw': yaw, 'roll': 0.}
            ob_bp = world.get_blueprint_library().find('static.prop.shoppingcart')
        elif anomaly_object_type == 5:  # Lying tree
            parameters = {'z': 0., 'pitch': 0., 'yaw': yaw, 'roll': 90.}
            ob_bp = world.get_blueprint_library().find('static.prop.tree')
        elif anomaly_object_type == 6:  # Lying tree
            parameters = {'z': 3., 'pitch': 0., 'yaw': yaw, 'roll': 0.}
            ob_bp = world.get_blueprint_library().find('static.prop.lod')
        elif anomaly_object_type == 7:  # Lying tree
            parameters = {'z': 0., 'pitch': 0., 'yaw': yaw, 'roll': 0.}
            ob_bp = world.get_blueprint_library().find('static.prop.swingcouch')
        elif anomaly_object_type == 8: # Secfence 3
            parameters = {'z': 0., 'pitch': 0., 'yaw': yaw, 'roll': 0.}
            ob_bp = world.get_blueprint_library().find('static.prop.secfence')
        elif anomaly_object_type == 9: # Secfence 1
            parameters = {'z': 0., 'pitch': 0., 'yaw': yaw, 'roll': 0.}
            ob_bp = world.get_blueprint_library().find('static.prop.secfence1')
        elif anomaly_object_type == 10: # Secfence 2
            parameters = {'z': 0, 'pitch': 90., 'yaw': yaw, 'roll': 0.}
            ob_bp = world.get_blueprint_library().find('static.prop.secfence2')
        elif anomaly_object_type == 11: # Guardrail
            parameters = {'z': 0., 'pitch': 0., 'yaw': yaw, 'roll': 0.}
            ob_bp = world.get_blueprint_library().find('static.prop.guardrail')
        elif anomaly_object_type == 12: # Ball (with issues)
            parameters = {'z': 0., 'pitch': 0., 'yaw': yaw, 'roll': 0.}
            ob_bp = world.get_blueprint_library().find('static.prop.ball')
        elif anomaly_object_type == 13: # Water drums
            parameters = {'z': 0., 'pitch': 0., 'yaw': yaw, 'roll': 0.}
            ob_bp = world.get_blueprint_library().find('static.prop.waterdrums')
        elif anomaly_object_type == 14: # Plastic Chair
            parameters = {'z': 0., 'pitch': 0., 'yaw': yaw, 'roll': 0.}
            ob_bp = world.get_blueprint_library().find('static.prop.plasticchair')
        elif anomaly_object_type == 15: # Plastic Table
            parameters = {'z': 0., 'pitch': 0., 'yaw': yaw, 'roll': 0.}
            ob_bp = world.get_blueprint_library().find('static.prop.plastictable')
        print('debug!!', spawn_point.rotation.yaw)
        if spawn_point.rotation.yaw < 45 and spawn_point.rotation.yaw > -45:
            ob = world.spawn_actor(
                ob_bp,
                carla.Transform(
                    carla.Location(x=spawn_point.location.x + anomaly_distance, y=spawn_point.location.y + random.random() * 2 - 1 - calib,
                                   z=parameters['z']),
                    carla.Rotation(pitch=parameters['pitch'], yaw=parameters['yaw'], roll=parameters['roll']),
                ))
        elif spawn_point.rotation.yaw > 45 and spawn_point.rotation.yaw < 135:
            ob = world.spawn_actor(
                ob_bp,
                carla.Transform(
                    carla.Location(x=spawn_point.location.x + random.random() * 2 - 1 - calib, y=spawn_point.location.y + anomaly_distance,
                                   z=parameters['z']),
                    carla.Rotation(pitch=parameters['pitch'], yaw=parameters['yaw'], roll=parameters['roll']),
                ))
        elif spawn_point.rotation.yaw > 135 or spawn_point.rotation.yaw < -135:
            ob = world.spawn_actor(
                ob_bp,
                carla.Transform(
                    carla.Location(x=spawn_point.location.x - anomaly_distance, y=spawn_point.location.y + random.random() * 2 - 1 + calib,
                                   z=parameters['z']),
                    carla.Rotation(pitch=parameters['pitch'], yaw=parameters['yaw'], roll=parameters['roll']),
                ))
        elif spawn_point.rotation.yaw < -45 and spawn_point.rotation.yaw > -135:
            ob = world.spawn_actor(
                ob_bp,
                carla.Transform(
                    carla.Location(x=spawn_point.location.x + random.random() * 2 - 1 + calib, y=spawn_point.location.y - anomaly_distance,
                                   z=parameters['z']),
                    carla.Rotation(pitch=parameters['pitch'], yaw=parameters['yaw'], roll=parameters['roll']),
                ))
        ob_list.append(ob)

        if args.no_rendering:
            settings.no_rendering_mode = True
        world.apply_settings(settings)

        blueprints = get_actor_blueprints(world, args.filterv, args.generationv)
        blueprintsWalkers = get_actor_blueprints(world, args.filterw, args.generationw)

        if args.safe:
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith('microlino')]
            blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
            blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('t2')]
            blueprints = [x for x in blueprints if not x.id.endswith('sprinter')]
            blueprints = [x for x in blueprints if not x.id.endswith('firetruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('ambulance')]

        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if args.number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif args.number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, args.number_of_vehicles, number_of_spawn_points)
            args.number_of_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        hero = args.hero
        for n, transform in enumerate(spawn_points):
            if n >= args.number_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            if hero:
                blueprint.set_attribute('role_name', 'hero')
                hero = False
            else:
                blueprint.set_attribute('role_name', 'autopilot')

            # spawn the cars and set their autopilot and light state all together
            batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)

        # Set automatic vehicle lights update if specified
        if args.car_lights_on:
            all_vehicle_actors = world.get_actors(vehicles_list)
            for actor in all_vehicle_actors:
                traffic_manager.update_vehicle_lights(actor, True)

        # -------------
        # Spawn Walkers
        # -------------
        # some settings
        percentagePedestriansRunning = 0.2  # how many pedestrians will run
        percentagePedestriansCrossing = 0.5  # how many pedestrians will walk through the road
        if args.seedw:
            world.set_pedestrians_seed(args.seedw)
            random.seed(args.seedw)
        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(args.number_of_walkers):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list[i]["con"] = results[i].actor_id
        # 4. we put together the walkers and controllers id to get the objects from their id
        for i in range(len(walkers_list)):
            all_id.append(walkers_list[i]["con"])
            all_id.append(walkers_list[i]["id"])
        all_actors = world.get_actors(all_id)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        # if args.asynch or not synchronous_master:
        #     world.wait_for_tick()
        # else:
        #     world.tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(world.get_random_location_from_navigation())
            # max speed
            all_actors[i].set_max_speed(float(walker_speed[int(i / 2)]))

        print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(vehicles_list), len(walkers_list)))

        traffic_list = world.get_actors().filter('*traffic_light*')
        for tf in traffic_list:
            tf.set_state(carla.TrafficLightState.Green)
            tf.freeze(True)

        # Example of how to use Traffic Manager parameters
        traffic_manager.global_percentage_speed_difference(30.0)

        #Simulation loop
        call_exit = False
        time_init_sim = timer.time()
        for i in tqdm(range(400)):
            # Carla Tick
            if args.sync:
                world.tick()
            else:
                world.wait_for_tick()

            # Render received data
            display_manager.render()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    call_exit = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == K_ESCAPE or event.key == K_q:
                        call_exit = True
                        break

            if call_exit:
                break

    finally:
        if display_manager:
            display_manager.destroy()

        if not args.asynch and synchronous_master:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.no_rendering_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(all_id), 2):
            all_actors[i].stop()

        client.apply_batch([carla.command.DestroyActor(x) for x in all_id])
        client.apply_batch([carla.command.DestroyActor(x) for x in ob_list])

        world.apply_settings(original_settings)


def get_actor_blueprints(world, filter, generation):
    # embed()
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []


def main():
    argparser = argparse.ArgumentParser(description='CARLA Sensor tutorial')
    argparser.add_argument('--host', metavar='H', default='localhost', help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    argparser.add_argument('--sync', action='store_true', help='Synchronous mode execution')
    argparser.add_argument('--async', dest='sync', action='store_false', help='Asynchronous mode execution')
    argparser.set_defaults(sync=True)
    argparser.add_argument('--res', metavar='WIDTHxHEIGHT', default='640x360', help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--weather',
        metavar='WEATHER',
        default=1,
        # weather list: []
        help='weather (default: rainy day)')

    argparser.add_argument('-n', '--number-of-vehicles', metavar='N', default=0, type=int, help='Number of vehicles (default: 10)')
    argparser.add_argument('-w', '--number-of-walkers', metavar='W', default=0, type=int, help='Number of walkers (default: 10)')
    argparser.add_argument('--safe', action='store_true', help='Avoid spawning vehicles prone to accidents')
    argparser.add_argument('--filterv', metavar='PATTERN', default='vehicle.*', help='Filter vehicle model (default: "vehicle.*")')
    argparser.add_argument('--generationv',
                           metavar='G',
                           default='All',
                           help='restrict to certain vehicle generation (values: "1","2","All" - default: "All")')
    argparser.add_argument('--filterw',
                           metavar='PATTERN',
                           default='walker.pedestrian.*',
                           help='Filter pedestrian type (default: "walker.pedestrian.*")')
    argparser.add_argument('--generationw',
                           metavar='G',
                           default='2',
                           help='restrict to certain pedestrian generation (values: "1","2","All" - default: "2")')
    argparser.add_argument('--tm-port', metavar='P', default=8000, type=int, help='Port to communicate with TM (default: 8000)')
    argparser.add_argument('--asynch', action='store_true', help='Activate asynchronous mode execution')
    argparser.add_argument('--hybrid', action='store_true', help='Activate hybrid mode for Traffic Manager')
    argparser.add_argument('-s', '--seed', metavar='S', type=int, help='Set random device seed and deterministic mode for Traffic Manager')
    argparser.add_argument('--seedw', metavar='S', default=0, type=int, help='Set the seed for pedestrians module')
    argparser.add_argument('--car-lights-on', action='store_true', default=False, help='Enable automatic car light management')
    argparser.add_argument('--hero', action='store_true', default=False, help='Set one of the vehicles as hero')
    argparser.add_argument('--respawn', action='store_true', default=False, help='Automatically respawn dormant vehicles (only in large maps)')
    argparser.add_argument('--no-rendering', action='store_true', default=False, help='Activate no rendering mode')
    argparser.add_argument('--file-dir', default='test', help='IP of the host server (default: 127.0.0.1)')

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    try:
        # while True:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)

        run_simulation(args, client)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
