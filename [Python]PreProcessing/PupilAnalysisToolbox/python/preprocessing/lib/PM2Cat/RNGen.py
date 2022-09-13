import math

#
#                    A progtram purely in Python
#                               of
#        Pierre L'Ecuyer's algorithm of random number generation
#
#                                       Yasuharu Okamoto,2016.09
#
#        cf. R. L. Eubank and A. Kupresanin (2012). "Statistical computing in C++ and R".
#

class RNp2to191:
    c209 = 2**32 - 209
    c22853 = 2**32 - 22853
  
    def __init__(self,*t):
        if (len(t) == 0):
            self.x10 = 3
            self.x11 = 2
            self.x12 = 1
            self.x20 = 6
            self.x21 = 5
            self.x22 = 4
        else:
            if (len(t) != 6):
                print("len(t) = %d"%(len(t)))
                raise "Number of parameters Error !"
            else:
                self.x10 = t[0]
                self.x11 = t[1]
                self.x12 = t[2]
                self.x20 = t[3]
                self.x21 = t[4]
                self.x22 = t[5]

    #
    #       Uniform distribution, 0 < uni() < 1
    #
    def uni(self):
        tx1 = (1403580 * self.x11 - 810728 * self.x12) % RNp2to191.c209
        self.x12 = self.x11
        self.x11 = self.x10
        self.x10 = tx1
        tx2 = (527612 * self.x20 - 1370589 * self.x22) % RNp2to191.c22853
        self.x22 = self.x21
        self.x21 = self.x20
        self.x20 = tx2

        z = (self.x10 - self.x20) % RNp2to191.c209
        if (z > 0):
            return z / (RNp2to191.c209 + 1.0)
        else:
            return RNp2t0191.c209 / (RNp2to191.c209 + 1.0)

    #
    #       Jump over 2**n numbers to be generated
    #
    def jump(self, n):
        if (n > 0):
            A = [[0, 1403580, -810728],
                 [1, 0, 0],
                 [0, 1, 0]]
            B = [[527612, 0, -1370589],
                 [1, 0, 0],
                 [0, 1, 0]]
            C = [[0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]]
            for i in range(0, n):
                for j in range(0, 3):
                    for k in range(0, 3):
                        tv = 0
                        for h in range(0, 3):
                            tv = tv + A[j][h] * A[h][k]
                        C[j][k] = tv
                for j in range(0, 3):
                    for k in range(0, 3):
                        A[j][k] = C[j][k] % RNp2to191.c209
                        
                for j in range(0, 3):
                    for k in range(0, 3):
                        tv = 0
                        for h in range(0, 3):
                            tv = tv + B[j][h] * B[h][k]
                        C[j][k] = tv
                for j in range(0, 3):
                    for k in range(0, 3):
                        B[j][k] = C[j][k] % RNp2to191.c22853

            t0 = A[0][0] * self.x10 + A[0][1] * self.x11 + A[0][2] * self.x12
            t1 = A[1][0] * self.x10 + A[1][1] * self.x11 + A[1][2] * self.x12
            t2 = A[2][0] * self.x10 + A[2][1] * self.x11 + A[2][2] * self.x12
            self.x10 = t0 % RNp2to191.c209
            self.x11 = t1 % RNp2to191.c209
            self.x12 = t2 % RNp2to191.c209
            t0 = B[0][0] * self.x20 + B[0][1] * self.x21 + B[0][2] * self.x22
            t1 = B[1][0] * self.x20 + B[1][1] * self.x21 + B[1][2] * self.x22
            t2 = B[2][0] * self.x20 + B[2][1] * self.x21 + B[2][2] * self.x22
            self.x20 = t0 % RNp2to191.c22853
            self.x21 = t1 % RNp2to191.c22853
            self.x22 = t2 % RNp2to191.c22853
             
    #
    #             Standard Normal Distribution
    #       Rejection polar method for normal variate
    #
    def normal(self):
        w = 0.0
        v1 = 0.0
        while True:
            v1 = 2.0 * self.uni() - 1.0
            v2 = 2.0 * self.uni() - 1.0
            w = (v1 ** 2.0) + (v2 ** 2.0)
            if ((w < 1.0) and (w > 0.0)):
                break
        c = math.sqrt(-2.0 * math.log(w) / w)

        return c * v1
    

    #
    #       Normal Distribution of mean m  and standard deviation s
    #
    def normalMS(self, m, s):
        return s * self.normal() + m


    # 
    #       A Pair of Independent Random Numbers of 
    #          the Standard Normal Distribution 
    # 
    def normalPair(self):
        w = 0.0
        v1 = 0.0
        v2 = 0.0
        while True:
            v1 = 2.0 * self.uni() - 1.0
            v2 = 2.0 * self.uni() - 1.0
            w = (v1 ** 2.0) + (v2 ** 2.0)
            if ((w < 1.0) and (w > 0.0)):
                break
        c = math.sqrt(-2.0 * math.log(w) / w)

        return c * v1, c * v2

