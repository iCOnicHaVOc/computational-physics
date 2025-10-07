from My2lib import *
import numpy as np
# Name = Aditya Raj, Roll no- 2311013
# Question 1

x = 0.1
c = 3.7
k= 9
X=[]  # list to store i + k th number from list l
def random(x,c): # def to generate random numbers
     l = [] # list to store generated random numbers
     oi = []
     for i in range (0,1000):
         oi.append(i)
         x = c*x*(1-x)
         l.append(x)
     return l

# y = random(x,c) # Calling the function
# for i in range (0,len(l)-k): # appending i+k th element from list l
#     z=l[i+k]
#     X.append(z)

# print(l)
# print(oi)
# print(len(oi))
# print(len(l))
# for storing random numbers in a txt file
# with open(r'P346\Assign1a\numSTORE.txt','w') as file:
#             for x in l:
#                    file.write(f"{str(x)}\n")

#for reading 
# with open(r'P346\Assign1a\numSTORE.txt','r') as file:


# # as list l is k elements bigger than X so I trim l to make it as same size as X 
# l_x = l[:-k]
# Plot(l_x,X)
# Plot(oi,l)

#Question 2
# a=1103515245
# e=12345
# m=32768
# j = 10
# k0 = 5
# Rand_y=[]

# call, f = LCG(10,500)

# for i in range (0,len(call)-k0): # appending i+k th element from list l
#     z=call[i+k0]
#     Rand_y.append(z)

# print(call)
# Plot(f,call)
# Rand_x = call[:-k0]
# Plot(Rand_x,Rand_y)

#Question 3

P_x , lda= LCG(17,5000)
P_y , lda = LCG(13,5000)

def check(x_list,y_list):
    Pi_n =[]
    total_points = len(x_list)
    inside_pi=0
    for point_num, (x, y) in enumerate(zip(x_list, y_list), start=1):
        if x**2 + y**2 <= 1:
            inside_pi += 1
        pi_est = 4 * (inside_pi / point_num)
        Pi_n.append(pi_est)
    return Pi_n

p = check(P_x,P_y)

p_avg = p[len(p)//2:]
print(len(p_avg))
avg = sum(p_avg)/len(p_avg)
print(avg)
# Plot(lda, p)

# Question 4
# import math

# seed = 2.1

# uniform_nums = LCG(seed)

# exp_nums = [-math.log(u) for u in uniform_nums if u > 0]  

# plt.hist(exp_nums, bins=100, density=True, alpha=0.7, color='skyblue', edgecolor='black')
# plt.xlabel("x")
# plt.ylabel("Probability Density")
# plt.title("Exponential Distribution from LCG ")
# plt.grid(True)
# plt.show()






