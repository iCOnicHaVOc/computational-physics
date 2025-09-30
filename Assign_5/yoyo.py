import math
def fun2(x):
    return math.log(x/2) - math.sin(5*x/2)


def root_bisection(fx,a,b,delta,eps,beta):

    if fx(a)*fx(b) > 0:
        print("The root is not bracketed, i'll do it for ya")
        fa,fb=fx(a),fx(b)
        s= 0
        while fa*fb >0 :
            if abs(fa) < abs(fb):
                a = a - beta*(b-a)
                fa = fx(a)
            if abs(fa) > abs(fb):
                b = b + beta*(b-a)
                fb = fx(b)
            s+=1
        a0,b0 = a,b
        print('Roots lie in the interval','[',a0,',',b0,'] and is reached in',s,'steps')
        return root_bisection(fx,a0,b0,delta,eps,beta)
    print(fx(a)*fx(b))
    steps = 0
    while True:
        c= (a+b)/2.0
        fc = fx(c)

        if abs(fc) < delta or abs(b-a) < eps:
            print('Checking Value of f(c) is',fc)
            return c,steps
        if fc*fx(a) < 0:
            b = c
        else:
            a = c
        steps+=1

x,y = root_bisection(fun2,5,6,1e-6,1e-6,0.2)
print("By bisection- Root (c) = ", x, 'Reached in',y, 'Steps')
