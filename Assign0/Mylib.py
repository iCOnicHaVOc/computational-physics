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
