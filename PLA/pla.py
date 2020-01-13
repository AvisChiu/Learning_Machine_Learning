import matplotlib as plt
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

def check_error(w, dataset):

    print("this is the episod...............")
    result = None
    error = 0
    for x, s in dataset:
        x = np.array(x)
        if int(np.sign(w.T.dot(x))) != s:

            result = x,s
            print("the w is:", w)
            print("the x is:", x)
            print("the dot is: ", np.sign(w.T.dot(x)))
            print("the s is:", s)
            print("the wrong is: ",result)
            error += 1

        if int(np.sign(w.T.dot(x))) == s:
            
            print("the w is:", w)
            print("the x is:", x)
            print("the dot is: ", np.sign(w.T.dot(x)))
            print("the s is:", s)
            print("the right is~!!!!!!!!!!!!!!!!!!!: ",result)

    print("total wrong: ",error)
    print("===========================")
    return result

def pla(dataset):
    w = np.zeros(3)
    while check_error(w,dataset) is not None:
        x,s = check_error(w,dataset)
        w += s*x
        print("the update is ~~~~~~~~~: ",w)
    return w

if __name__ == "__main__":
    
    w = pla(dataset)


# print(w)
# l = np.linspace(-2,2, num=6)
# print(l)