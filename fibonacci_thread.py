import threading

def add(x,y):
    print(x + y)

def main():
    a,b = 0,1
    print(a,b,end=' ')
    
    for i in range(8):
        t1 = threading.Thread(target=add, args=(a,b))
        t1.start()
        res = t1.join()
        a,b=b,res
if __name__ == "__main__":
    main()
