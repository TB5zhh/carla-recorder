from argparse import Namespace
from pathlib import Path
import sys, os, time
import itertools
from multiprocessing import Process
from ad_generator import main


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
        process = Process(target=main, args=(sys.argv, ))
        process.start()
        process.join()
        kill_server()  # Kill server!


def my_iter():

    args = sys.argv

    import random
    spawn_point = [20, 50, 90] # random.sample(range(155), 10) # 5
    cloudiness = [0] # 2
    precipitation = [0] # 2
    fog_density = [0] # 2, []
    sun_altitude_angle = [-10] # 3, [70]

    # cloudiness = precipitation = fog_density = [0, ]
    # sun_altitude_angle = [70, ]

    n = [60, ] # 1
    w  = [100, ] # 1

    Targs = Namespace()

    for config in itertools.product(spawn_point, cloudiness, precipitation, fog_density, sun_altitude_angle, n, w):
        Targs.spawn_point = config[0]
        
        if config[2] == 0:
            Targs.cloudiness, Targs.precipitation, Targs.fog_density = 0, 0, 0
        elif config[2] == 1:
            Targs.cloudiness, Targs.precipitation, Targs.fog_density = 0, 0, 50
        elif config[2] == 2:
            Targs.cloudiness, Targs.precipitation, Targs.fog_density = 0, 90, 20

        
        # Targs.cloudiness = config[1]
        # Targs.precipitation = config[2]
        # Targs.fog_density = config[3]
        Targs.sun_altitude_angle = config[4]
        Targs.number_of_vehicles = config[5]
        Targs.number_of_walkers = config[6]
        Targs.car_lights_on = True

        Targs.file_dir = f"/media/ubuntu/c7wyyds/Carla/sp{config[0]}-cloud{config[1]}-pre{config[2]}-fog{config[3]}-sun{config[4]}-n{config[5]}-v{config[6]}"

        # 5 * 8 * 3 = 120
        # 120 * 600M = 0.12 * 600 G = 72 G
        process = Process(target=main, args=(args, Targs))
        process.start()
        process.join()
        kill_server()  # Kill server!

    
if __name__ == "__main__":
    my_iter()
