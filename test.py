# coding: utf-8

import unittest
from raytracer2d import *
import numpy as np 


class TestRay(unittest.TestCase):
    def test_ray(self):

        ray = Ray(Vector2D(0, 0), Vector2D(2, 0), 550)
        self.assertEqual(Vector2D(1, 0), ray.d)

class TestVector2D(unittest.TestCase):

    def test_operator(self):
        v1 = Vector2D(0, 1)
        v2 = Vector2D(2, 3)

        self.assertTrue(v1 == Vector2D(0., 1.))
        self.assertEqual(v1 + v2, Vector2D(2, 4))
        self.assertEqual(v1 - v2, Vector2D(-2, -2))
        self.assertEqual(3 * v1, Vector2D(0, 3))
        self.assertEqual(v1 / 5, Vector2D(0, 1/5))
        self.assertEqual(-v2, v2*-1)
        self.assertEqual(v1@v2, 3)
        self.assertEqual(v2.length(), np.sqrt(2**2 + 3**2))

class TestSurface(unittest.TestCase):

    def test_plane(self):
        plane = Plane(Vector2D(0, 0), Vector2D(0, 1), 10)
        ray1 = Ray(Vector2D(0, 1), Vector2D(0, -1))
        ray2 = Ray(Vector2D(0, 1), Vector2D(0, 1))
        ray3 = Ray(Vector2D(0, 1), Vector2D(1, 0))

        self.assertEqual(plane.intersect(ray1), 1)
        self.assertEqual(plane.intersect(ray2), -1)
        self.assertEqual(plane.intersect(ray3), None)
        self.assertEqual(plane.get_coordinate(1), Vector2D(5, 0))
        self.assertEqual(plane.get_coordinate(-1), Vector2D(-5, 0))



class TestMaterial(unittest.TestCase):
    def test_fresnel(self):

        fresnel = Fresnel(n1=1.0, n2=1.5)

        ray = Ray(Vector2D(0, 0), Vector2D(0, -1))
        normal = Vector2D(0, 2)
        with self.assertRaises(Exception):
            fresnel.interact(ray, normal)    
        
        #normal incident
        ray = Ray(Vector2D(0, 0), Vector2D(0, -1))
        normal = Vector2D(0, 1)
        (t, ray_t), (r, ray_r) = fresnel.compute_output_ray(ray, normal)

        self.assertAlmostEqual(t+r, 1)
        self.assertAlmostEqual(r, 0.04)
        self.assertEqual(ray.o, ray_r.o)
        self.assertEqual(ray.o, ray_t.o)
        self.assertEqual(ray_r.d, Vector2D(0, 1))
        self.assertEqual(ray_t.d, Vector2D(0, -1))


        #refraction. incidnet angl = 30 deg
        ray = Ray(Vector2D(0, 0), Vector2D(1, -np.sqrt(3)))
        normal = Vector2D(0, 1)
        (t, ray_t), (r, ray_r) = fresnel.compute_output_ray(ray, normal)
        
        self.assertAlmostEqual(t+r, 1)
        self.assertAlmostEqual(ray_t.d @ -normal, np.cos(19.471220*np.pi / 180))

        #total reflection
        fresnel = Fresnel(n1=1, n2=1.5)
        ray = Ray(Vector2D(0, 0), Vector2D(np.sqrt(3), 1))
        normal = Vector2D(0, 1)
        (t, ray_t), (r, ray_r) = fresnel.compute_output_ray(ray, normal)

        self.assertAlmostEqual(t+r, 1)
        self.assertEqual(r, 1)
        self.assertAlmostEqual(ray_r.d, Vector2D(np.sqrt(3), -1) / 2)
    
    def test_mirror(self):
        mirror = Mirror()

        ray = Ray(Vector2D(0, 0), Vector2D(1, -1))
        normal = Vector2D(0, 1)

        out_ray = mirror.interact(ray, normal)

        self.assertAlmostEqual(out_ray.d, Vector2D(1, 1).normalize())


class TestSource(unittest.TestCase):
    def test_parallel_source(self):
        plane = Plane(Vector2D(0, 0), Vector2D(0, 1), 1)
        source = ParalleSource(plane, wl=550)
        ray = source.emit()
        self.assertEqual(ray.d, plane.n)
    
    def test_point_source(self):
        source = PointSource(Vector2D(0, 1), wl=550)
        ray = source.emit()
        self.assertEqual(ray.o, Vector2D(0, 1))


class TestCamera(unittest.TestCase):

    def setUp(self):
        o = Vector2D(0, 0)
        d = Vector2D(1, 0)
        f = 10
        fno = 10
        bf = 2*f
        n_pixel = 11
        pixel_size = 1
        n_sub_pupil = 11
        diameter = f / fno
        sub_pupil_size = diameter / n_sub_pupil
        self.camera = Camera(o, d, f, fno, bf, n_pixel, pixel_size, n_sub_pupil) 


    def test_basic_functions(self):
        camera = self.camera
        sub_pupil_size = camera.pupil.length / camera.n_sub_pupil

        self.assertEqual(camera.pupil.o, camera.o+camera.bf*camera.d)
        self.assertEqual(camera.pupil.n, camera.d)
        self.assertEqual(camera.sensor.n, camera.d)
        self.assertEqual(camera.pupil.length, camera.f/camera.fno)
        self.assertEqual(camera.sensor.length, camera.pixel_size * camera.n_pixel)

        self.assertEqual(camera.get_pixel_coordinate(camera.n_pixel//2), camera.o)
        self.assertEqual(camera.get_pupil_coordinate(camera.n_pixel//2), camera.pupil.o)

        self.assertAlmostEqual(camera.get_pixel_coordinate(camera.n_pixel//2-1), 
                               camera.o + camera.pixel_size * camera.d.orthogonal())
        self.assertAlmostEqual(camera.get_pupil_coordinate(camera.n_pixel//2-3),
                               camera.pupil.o + 3 * sub_pupil_size * camera.d.orthogonal())
        

    def test_conjugate(self):
        camera = self.camera

        p = Vector2D(0, 1)
        self.assertAlmostEqual(camera.conjugate(p), Vector2D(4*camera.f, -1))

        camera.bf = camera.f
        p = Vector2D(0, 0)
        self.assertAlmostEqual(camera.conjugate(p), camera.pupil.o+Vector2D(INF, 0),)


    def test_refract(self):

        camera = self.camera

        p_pixel = camera.get_pixel_coordinate(camera.n_pixel // 2)
        p_pupil = camera.get_pupil_coordinate(4)
        plane = Plane(Vector2D(4*camera.f, 0), Vector2D(-1, 0), length=10)
        ray_out = camera.refract(p_pixel, p_pupil, wl=None)
        ray_out = ray_out.travel(plane.intersect(ray_out))
        self.assertAlmostEqual(ray_out.o, Vector2D(4*camera.f, 0))

        p_pixel2 = p_pixel + Vector2D(0, 1)
        ray_out = camera.refract(p_pixel2, p_pupil, wl=None)
        ray_out = ray_out.travel(plane.intersect(ray_out))
        self.assertAlmostEqual(ray_out.o, Vector2D(4*camera.f, -1))





        




        



if __name__ == "__main__":
    unittest.main()



