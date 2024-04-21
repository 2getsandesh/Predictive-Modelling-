import numpy as np
import threading

def minpool(i,j):
    j.append(np.min(i))

def maxpool(i,j):
    j.append(np.max(i))


def main():
    m = np.array([[1,2,3,4],
             [5,6,7,8],
             [9,10,11,12],
             [13,14,15,16]])

    subarray = [
        m[:2,:2],
        m[:2,2:],
        m[2:,:2],
        m[2:,2:]
    ]

    minpool_res = []
    maxpool_res = []

    for i in subarray:
        t1 = threading.Thread(target=minpool, args=(i,minpool_res))
        t2 = threading.Thread(target=maxpool, args=(i,maxpool_res))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

    print(minpool_res)
    print(maxpool_res)

if __name__ == "__main__":
    main()
