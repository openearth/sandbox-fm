# This script will start a loop that writes every second (fast loop). 
# However, that loop requires a value set by a loop that only updates every 5 seconds. 
# A pipe and process is used to make both communicate.

import threading
from multiprocessing import Process, Queue, Pipe
import time
import atexit


# Create new log file
outputfile = 'log.txt'
with open(outputfile, 'w') as f:
    f.write('Starting log\n')


# Start the slow loop
def update_queue_slowly(a,):
    count = 0 
    while True:
        with open(outputfile, 'a') as f:
            f.write('Slow: {} \n'.format(count))
            a.send(count)
        count += 1
        time.sleep(5)


if __name__ == '__main__':
    # Create pipe from a to b
    a,b = Pipe()
    p = Process(target=update_queue_slowly, args=(a, ))
    p.start()

    # If script is aborted, also kill the subprocess
    atexit.register(p.kill())

    value=0

    # Start the fast loop
    while True:
        time.sleep(1)

        # Get value from pipe, but only if new data is available
        if b.poll():
            value = b.recv()

        with open(outputfile, 'a') as f:
            f.write('Fast: {} \n'.format(value))            
    p.terminate()

