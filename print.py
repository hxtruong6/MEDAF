import time


# print 1-1000 sleep 1 second each
for i in range(1, 10001):
    print(i)
    print(f"sleep 1 second {i}")
    time.sleep(1)
