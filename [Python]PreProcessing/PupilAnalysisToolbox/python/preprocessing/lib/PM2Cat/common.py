import math

def cum_normalKC( z ):
    """	
	Kerridge and Cook(1976, Biometrika, 63, 401-403)'s Algorithm
	
    """
    def normal_0_to_z( z ):
        p = 0.0
        if z > 6.8:
            zz = z * z
            d = zz + 3 - 1 / (0.22 * zz + 0.704)
            m = 1 - 1.0 / d
            p = 0.5 - math.exp(-0.5 * zz) * m / (z * math.sqrt(2 * math.pi))
        else:
            zz4 = 0.25 * z * z
            n2 = 2.0
            theta2 = 1.0
            theta1 = zz4
            p = 1.0
            p_prev = 0.0

            while True:
                theta2 = zz4 * (theta1 - theta2) /n2
                n2 += 1.0
                p += theta2 / n2
                if p == p_prev:
                    break
                theta1 = zz4 * (theta2 - theta1) / n2
                n2 += 1.0
                p_prev = p

            p = z * math.exp(-0.5 * zz4) * p / math.sqrt(2.0 * math.pi)

        return p

    if z >= 0.0:
        if z > 0.0:
            v = 0.5 + normal_0_to_z( z )
            if v > 1.0:
                return 1.0
            else:
                return v
        else:
            return 0.5
    else:
        v = 0.5 - normal_0_to_z( -z )
        if v < 0.0:
            return 0.0
        else:
            return v
        

