import sys, os, time
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

while True:
    process = Process(target=main, args=(sys.argv, ))
    process.start()
    process.join()
    kill_server() # Kill server!