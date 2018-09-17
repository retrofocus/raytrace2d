# coding: utf-8

import numpy as np


INF = 1e16

def rand_pm1():
    return np.random.rand() *  2 - 1



class Ray:
    def __init__(self, o, d, wl=None):
        """Ray class
        
        Arguments:
            o {numpy array} -- [origine]
            d {numpy array} -- [direction. automatically normalized if lenght is not 1.]
            wl {float} -- wavelength in [nm]
        """

        self.o = o
        if d.length() != 1:
            self.d = d.normalize()
        else:
            self.d = d
        
        self.wl = wl

    def travel(self, t):
        return Ray(self.o + t * self.d, self.d)
    


class Vector2D:
    __slots__ = ['x', 'y']
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return Vector2D(self.x+other.x, self.y+other.y)
    
    def __sub__(self, other):
        return Vector2D(self.x-other.x, self.y-other.y)
    
    def __neg__(self):
        return Vector2D(-self.x, -self.y)
    
    def __mul__(self, a):
        if isinstance(a, Vector2D):
            raise TypeError("Vector2D cannot multiplied with Vector2D, only float.")
        return Vector2D(self.x * a, self.y * a)
    
    def __rmul__(self, a):
        return self * a
    
    def __truediv__(self, a):
        return Vector2D(self.x / a, self.y / a)

    def __matmul__(self, other):
        return self.x*other.x + self.y*other.y
    

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def normalize(self):
        return Vector2D(self.x, self.y) / self.length()
    
    def __repr__(self):
        return "x: %f, y:%f" % (self.x, self.y)
    
    def __abs__(self):
        return self.length()
    
    def length(self):
        return np.sqrt(self.x**2 + self.y**2)

    def orthogonal(self):
        return Vector2D(-self.y, self.x)






class Surface:
    """base class for objects.
    
    Raises:
        NotImplementedError -- [description]
    """
    
    def next(self, ray):
        raise NotImplementedError()


class Plane(Surface):
    def __init__(self, o, n, length):
        """
        
        Arguments:
            o {Vector2D} -- [origin]
            n {Vector2D} -- [normal vector]
            lenght {float} -- lenth of plane. edges are length / 2 from origin.
        """

        self.o = o
        self.n = n
        self.length = length
    
    def next(self, ray):
        raise NotImplementedError()
    
    def intersect(self, ray):
        """https://qiita.com/tmakimoto/items/2da05225633272ef935c
        if the ray intersects, returns the distance to the point.
        otherwise returns None.
        
        Arguments:
            ray {Ray} -- [Ray to intersect.]
        
        Returns:
            float -- [distance to the intersection]
        """
        p = ray.o
        u = ray.d
        q = self.o
        v = Vector2D(-self.n.y, self.n.x)
        n = self.n

        if n @ u == 0:
            return None

        s = n @ (q - p) / (n @ u)

        if abs(ray.travel(s).o - self.o) > self.length / 2:
            return None

        return s
    
    def get_coordinate(self, l):
        """returns the coordinate(Vector2D) which is specified by length from origin.
        
        Arguments:
            l {float} -- from -1 to 1.
        
        Returns:
            Vector2D -- correspoinding coordinate.
        """

        d = -self.n.orthogonal()
        
        v = self.o + l * self.length/2 * d

        return v


class Material:
    def interact(self, ray, normal):
        raise NotImplementedError()

class Lambert(Material):
    def __init__(self, r, t):
        """
            r {float} -- reflection ratio
            t {float} -- transmition ratio
        """

        self.r = r
        self.t = t
    
    def interact(self, ray, normal):
        """[summary]
        
        Arguments:
            ray {Ray} -- incident ray
        """

        sign = np.sign(ray @ normal)
        ortho = normal.orthogonal()

        if np.random.rand() < r:
            return -sign * np.random.rand() * normal + (2 * np.random.rand() - 1) * ortho
            
        else:
            return sing * np.random.rand() * normal + (2 * np.random.rand() - 1) * ortho

class Mirror(Material):
    def __init__(self):
        pass
    
    def interact(self, ray, normal):

        return Ray(ray.o, ray.d - 2 * normal * (ray.d @ normal), ray.wl) #reflected




class Fresnel(Material):
    def __init__(self, n1, n2):
        self.n1 = n1
        self.n2 = n2
    
    def compute_amplitude_reflection_coefficient(self, a, b, n1, n2):
        """compute reflection raio
        
        Arguments:
            a {float} -- incident angle in radians
            b {float} -- refracted angle in radians
            n1{float} -- refractive index of incident medium
            n2{float} -- refractive index of refracted medium
        
        Returns:
            [tuple] -- reflection ratio of (s, p)
        """
        rp = (n2 * np.cos(a) - n1 * np.cos(b)) / (n2 * np.cos(a) + n1 * np.cos(b))
        rs = (n1 * np.cos(a) - n2 * np.cos(b)) / (n1 * np.cos(a) + n2 * np.cos(b))

        return rs, rp

    
    def compute_amplitude_refraction_coefficient(self, a, b, n1, n2):
        """compute refration raio
        
        Arguments:
            a {float} -- incident angle in radians
            b {float} -- refracted angle in radians
            n1{float} -- refractive index of incident medium
            n2{float} -- refractive index of refracted medium
        
        Returns:
            [tuple] -- refraction ratio of (s, p)
        """
        tp = 2 * n1 * np.cos(a) / (n2 * np.cos(a) + n1 * np.cos(b))
        ts = 2 * n1 * np.cos(a) / (n1 * np.cos(a) + n2 * np.cos(b))

        return ts, tp
    
    def compute_output_ray(self, ray, n_vec):


        if n_vec.length() != 1:
            raise ValueError("input normal length must be 1.")

        #switching the normal based on relative direction of ray and n_vec.
        if ray.d @ n_vec < 0: 
            # ray travels from n1 to n2
            n1 = self.n1
            n2 = self.n2
            normal = -n_vec

        else:
            #from n2 to n1
            n1 = self.n2
            n2 = self.n1
            normal = n_vec
        
        cos_a = ray.d @ normal
        ray_r = Ray(ray.o, ray.d - 2 * normal * cos_a, ray.wl) #reflected
        a = np.arccos(cos_a) # incident angle
        sin_b = n1 * np.sin(a) / n2
        if sin_b > 1:
            #total reflection
            return (0.0, ray), (1.0, ray_r)
        
        cos_b = np.sqrt(1 - (n1/n2)**2 * (1 - cos_a**2))
        #transimittd
        ray_t = Ray(ray.o, 
                    n1/n2*ray.d + normal*(cos_b - n1/n2*cos_a),
                    ray.wl)

        
        b = np.arcsin(sin_b) # refraction angle

        rs, rp = self.compute_amplitude_reflection_coefficient(a, b, n1, n2)
        ts, tp = self.compute_amplitude_refraction_coefficient(a, b, n1, n2)

        t = n2 * np.cos(b) / (n1 * np.cos(a)) * (tp**2 + ts**2) / 2
        r = (rp**2 + rs**2) / 2

        return (t, ray_t), (r, ray_r)
    
    def interact(self, ray, n_vec):
        """Fresnel refraction and reflection.(see wikipedia)
        interation is supposed to occurs at ray origin.
       
        Arguments:
            ray {Ray} -- input ray.
            n_vec {Vector2D} -- surface normal. normal should be toward n2 to n1.
        
        Returns:
            Ray -- ray after interaction.
        """


        (t, ray_t), (r, ray_r) = self.compute_output_ray(ray, n_vec)
        if np.random.rand() < t: 
            #refraction
            return ray_r
        else:
            return ray_t


class Source:
    def emit(self):
        raise NotImplementedError()
    
    def radiance(self, ray):
        raise NotImplementedError()


class ParalleSource(Source):
    """Parallel light source that emit ray towards its normal vector.
    """

    def __init__(self, plane, wl):
        self.plane = plane
        self.wl = wl
    
    def emit(self):
        l = 2 * np.random.rand() - 1

        o = self.plane.get_coordinate(l)
        ray = Ray(o, self.plane.n, self.wl)
    
        return ray
    
    def radiance(self):
        pass

class PointSource(Source):
    def __init__(self, o, wl):
        self.o = o
        self.wl = wl
    
    def emit(self):
        theta = np.random.rand() * 2 * np.pi
        d = Vector2D(np.cos(theta), np.sin(theta))

        return Ray(self.o, d, self.wl)
    


class Camera:
    def __init__(self, o, d, f, fno, bf, n_pixel, pixel_size, n_sub_pupil):

        self.f = f
        self.fno = fno
        diameter = f / fno
        self._bf = bf

        self.sensor = Plane(o, d, n_pixel * pixel_size)
        self.pupil = Plane(o + bf*d, d, diameter)
        self.n_pixel = n_pixel
        self.pixel_size = pixel_size
        self.n_sub_pupil = n_sub_pupil

        self.lf = np.zeros((n_pixel, n_sub_pupil))

    
    @property
    def o(self):
        return self.sensor.o
    
    @o.setter
    def o(self, v):
        self.sensor.o = v
        self.pupil.o = v + self._bf * self.d
    
    @property
    def d(self):
        return self.sensor.n
    
    @d.setter
    def d(self, v):
        v = v.normalize()
        self.sensor.n = v
        self.pupil.n = v
    
    @property
    def bf(self):
        return self._bf
    
    @bf.setter
    def bf(self, v):
        self._bf = v
        self.pupil.o = self.o + self._bf * self.d
    

    def conjugate(self, p):
        """returns conjugate point of a given point.
        
        Arguments:
            p {Vector2D} -- point of which conjugate to be computed.
        
        Returns:
            Vector2D -- conjugate point of input point.
        """

        s = (p - self.pupil.o) @ self.d
        h = (p - self.pupil.o) @ self.d.orthogonal()

        sd_inv = (1/self.f + 1/s)
        if abs(sd_inv) < 1/INF:
            beta = 0
            sd = INF
        else:
            sd = 1 / sd_inv
            beta = sd / s

        hd = beta * h

        conj = self.pupil.o + sd * self.d + hd * self.d.orthogonal()

        return conj

    def focus(self, p):
        """move lens(change Bf) and focus at point p.
        
        Arguments:
            p {Vector2D} -- point to be focused
        """

        pass
        



    def refract(self, p_pixel, p_pupil, wl):
        conj = self.conjugate(p_pixel)
        
        return Ray(p_pupil, conj - p_pupil, wl=wl)



    def get_pupil_coordinate(self, i, random=False):
        """get pixel coordinate given its index
        
        Arguments:
            i {int} -- from 0 to n - 1
        
        Keyword Arguments:
            random {bool} -- if True, given a random point in the pixel (default: {False})
        
        Returns:
            Vector2D -- pixel coordinate
        """
        
        x = ((i+0.5) / self.n_sub_pupil - 0.5 ) * 2 # from -1 to 1
        
        v = self.pupil.get_coordinate(x)

        if random:
            sub_pupil_size = self.pupil.length / self.n_sub_pupil
            v +=  sub_pupil_size * rand_pm1()

        return v

    def get_pixel_coordinate(self, i, random=False):
        """get pixel coordinate given its index
        
        Arguments:
            i {int} -- from 0 to n - 1
        
        Keyword Arguments:
            random {bool} -- if True, given a random point in the pixel (default: {False})
        
        Returns:
            Vector2D -- pixel coordinate
        """
        
        x = ((i+0.5) / self.n_pixel - 0.5 ) * 2 # from -1 to 1
        
        v = self.sensor.get_coordinate(x)

        if random:
            v += self.pixel_size * rand_pm1()

        return v
    


class Tracer:
    def __init__(self, lights, camera, surfaces, seed=1):
        np.random.seed(seed)

        self.lights = lights
        self.camera = camera
        self.surfaces = surfaces

        
    

    
    
    

