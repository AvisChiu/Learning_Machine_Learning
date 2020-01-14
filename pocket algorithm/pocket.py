# import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np 

dataset = np.array([        # x0, x1, x2,  x0=1
    ((1,-0.4,0.3),-1),
    ((1,-0.3,0.1),-1),
    ((1,-0.2,0.4),-1),
    ((1,-0.1,0.1),-1),
    ((1,0.9,-0.5),1),
    ((1,0.7,-0.9),1),
    ((1,0.8,0.2),1),
    ((1,0.4,-0.6),1),
])


def check(w,dataset):
    count = 0
    for x, s in dataset:
        x = np.array(x)
        if int(np.sign(w.T.dot(x))) != s:
            count += 1 
            tmp.append((x,s))     
    return count

if __name__ == "__main__":
    
    tmp = []
    w = np.zeros(3)
    initial = check(w,dataset)

    for x, s in tmp:
        w = w + s*x
        tmp = []
        new = check(w, dataset)
        if new < initial:
            initial = new

    print(w)
    print(initial)

    
    
    fig = plt.figure()
 
    # numrows=1, numcols=1, fignum=1
    ax1 = fig.add_subplot(111)
 
    xx = list(filter(lambda d: d[1] == -1, dataset))
    ax1.scatter([x[0][1] for x in xx], [x[0][2] for x in xx],
                s=100, c='b', marker="x", label='-1')
    oo = list(filter(lambda d: d[1] == 1, dataset))
    ax1.scatter([x[0][1] for x in oo], [x[0][2] for x in oo],
                s=100, c='r', marker="o", label='1')
    l = np.linspace(-2, 2)
 
    # w0 + w1x + w2y = 0
    # y = -w0/w2 - w1/w2 x
    if w[2]:
        a, b = -w[1] / w[2], -w[0] / w[2]
        ax1.plot(l, a * l + b, 'b-')
    else:
        ax1.plot([-w[0] / w[1]] * len(l), l, 'b-')
 
    plt.legend(loc='upper left', scatterpoints=1)
    plt.show()
    


