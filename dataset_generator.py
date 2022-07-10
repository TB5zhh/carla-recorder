from argparse import Namespace
import subprocess
from pathlib import Path
import sys, os, time
import itertools
from multiprocessing import Process

from matplotlib.style import available
from ad_generator import main

from config import *

def kill_server():
    try:
        for line in os.popen("ps | grep CarlaUE4"):
            if line:
                line = int(line.strip().split(' ')[0])
                os.kill(line, 11)
        time.sleep(5)
    except:
        pass


def run_forever():
    while True:
        process = Process(target=main, args=(sys.argv,))
        process.start()
        process.join()
        kill_server()  # Kill server!


def my_iter():

    args = sys.argv

    import random
    cloudiness = [0]  # 2
    precipitation = [0]  # 2
    fog_density = [0]  # 2, []
    sun_altitude_angle = [70]  # 3, [70]
    n = [30]  # 1
    w = [100]  # 1

    available_maps = ['Town10HD']
    spawn_point = [55]  # random.sample(range(155), 10) # 5

    anomaly_types = list(range(1))
    with_anomaly = [True]
    # cloudiness = precipitation = fog_density = [0, ]
    # sun_altitude_angle = [70, ]

    Targs = Namespace()

    for params in itertools.product(
            anomaly_types,
            available_maps,
            spawn_point,
            cloudiness,
            precipitation,
            fog_density,
            sun_altitude_angle,
            n,
            w,
            with_anomaly,
    ):
        (
            anomaly,
            map,
            sp,
            cloud,
            prec,
            fog,
            sun_angle,
            n_,
            w_,
            with_ano,
        ) = params
        Targs.map = map
        Targs.spawn_point = sp

        if prec == 0:
            Targs.cloudiness, Targs.precipitation, Targs.fog_density = 0, 0, 0
        elif prec == 1:
            Targs.cloudiness, Targs.precipitation, Targs.fog_density = 0, 0, 50
        elif prec == 2:
            Targs.cloudiness, Targs.precipitation, Targs.fog_density = 0, 90, 20

        # Targs.cloudiness = config[1]
        # Targs.precipitation = config[2]
        # Targs.fog_density = config[3]
        Targs.sun_altitude_angle = sun_angle
        Targs.number_of_vehicles = n_
        Targs.number_of_walkers = w_
        Targs.car_lights_on = False

        Targs.generate_anomaly = with_ano
        Targs.anomaly_type = anomaly
        Targs.seq_length = 400
        Targs.frame_interval = 1 / 40

        #Targs.file_dir = f"/media/ubuntu/c7wyyds/Carla/sp{config[0]}-cloud{config[1]}-pre{config[2]}-fog{config[3]}-sun{config[4]}-n{config[5]}-v{config[6]}"
        # Targs.file_dir = f"/home/ubuntu/tb5zhh/carla/recorder/anomaly_dataset_v0.1/{anomaly}/{map}_{sp}/{with_ano}"
        Targs.file_dir = f"/media/ubuntu/0997C4D95A4FEE10/anomaly_dataset_v0.1.1"
        # 5 * 8 * 3 = 120
        # 120 * 600M = 0.12 * 600 G = 72 G
        while True:
            server_process = subprocess.Popen(
                    CARLA_EXE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
            print(params)
            time.sleep(10)
            # try:
            process = Process(target=main, args=(args, Targs))
            process.start()
            process.join()
            kill_server()
            if process.exitcode == 0:
                break
            print('Retrying...')


if __name__ == "__main__":
    my_iter()
