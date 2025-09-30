class MyComplex:
    def __init__(self, real, imag=0.0):
        # Store the real and imaginary parts
        self.r = real
        self.i = imag

    def display_cmplx(self):
        # Print in the form a,bj without spaces
        print(self.r, ",", self.i, "j", sep="")

    def add_cmplx(self, c1, c2):
        # Add corresponding real and imaginary parts
        self.r = c1.r + c2.r
        self.i = c1.i + c2.i
        return MyComplex(self)

    def sub_cmplx(self, c1, c2):
        # Subtract corresponding real and imaginary parts
        self.r = c1.r - c2.r
        self.i = c1.i - c2.i
        return MyComplex(self)

    def mul_cmplx(self, c1, c2):
        # Multiply using (a+bi)(c+di) = (ac - bd) + (ad + bc)i
        self.r = c1.r * c2.r - c1.i * c2.i
        self.i = c1.i * c2.r + c1.r * c2.i
        return MyComplex(self)

    def mod_cmplx(self):
        # Modulus: sqrt(real^2 + imag^2) without NumPy
        return (self.r ** 2 + self.i ** 2) ** 0.5

# class randGEN():
#     def __init__(self):
#         self.rand= []
#         self.index=[]
#     def LCG(self,p,q,r,k):
#         for i in range (0,500): 
#             self.index.append(i)
#             k = (p*k+q)%r 
#             f = k/r
#             self.rand.append(f)
#         print(self.rand)
#         print(self.index)
#         return randGEN(self)
    
# THIS GIVE A SINGLE RANDOM NO. AS OUTPUT
# def LCG(p,q,r,k,n):
#     for i in range (0,n): 
#         k = (p*k+q)%r 
#         f = k/r
#     return f

# ploting the lists 
import matplotlib.pyplot as plt 
#define x and y , as a python list 
def Plot(x, y , title='Pi plot 2000 throws', xlabel='index', ylabel='Values of Pi',file_name='sample_plot.png'):
    plt.figure(figsize=(11, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)  
    plt.savefig(file_name)
    plt.show()

#GIVES A LIST WITH RANDOM NUMBERS 
def LCG(k,it):
    p=1103515245
    q=12345
    r=32768
    Rand = []
    index = []
    for i in range (0,it): 
        index.append(i)
        k = (p*k+q)%r 
        f = k/r
        Rand.append(f)
    return Rand , index


    
    
    