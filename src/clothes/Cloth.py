import taichi as ti
from abc import ABCMeta, abstractmethod
from src.objects.Object import Object

@ti.data_oriented
class Cloth(object, metaclass=ABCMeta):
    def __init__(self, V, POSITIONS, VELOCITIES, WEIGHT, indices, edges, tripairs, color, KS, KC, KB, KD, KL, DAMPING):
        self.V = V
        self.POSITIONS = POSITIONS
        self.VELOCITIES = VELOCITIES
        self.indices = indices
        self.edges = edges
        self.tripairs = tripairs
        self.color = color
        self.KS = KS
        self.KC = KC
        self.KB = KB
        self.KL = KL
        self.KD = KD
        self.DAMPING = DAMPING
        self.objs = None

        self.vertices = ti.Vector.field(3, float, self.V)
        self.x = ti.Vector.field(3, float, self.V)
        self.fixed = ti.field(int, self.V)
        self.nt = ti.field(int, self.V)

        for j in range(self.indices.shape[0]//3):
            vi1 = self.indices[j * 3 + 0]
            vi2 = self.indices[j * 3 + 1]
            vi3 = self.indices[j * 3 + 2]

            self.nt[vi1] += 1
            self.nt[vi2] += 1
            self.nt[vi3] += 1

        self.v = ti.Vector.field(3, float, self.V)
        self.dv = ti.Vector.field(3, float, self.V)
        self.p = ti.Vector.field(3, float, self.V)
        self.WEIGHT = WEIGHT

        self.reset_pos_and_vel()

    """
    sets pos and vel to initial pos
    """
    def reset_pos_and_vel(self):
        for i in range(self.V):
            self.x[i] = self.POSITIONS[i]
        
        for i in range(self.V):
            self.v[i] = self.VELOCITIES[i]


    """
    draws the cloth
    """
    def draw(self, scene):
        scene.mesh(self.x,
               indices=self.indices,
               color=self.color,
               two_sided=True)

    @ti.kernel
    def external_forces(self, G : ti.f32, wind: ti.template(), DT : ti.f32):
        for i in range(self.V):
            self.dv[i] = ti.Vector([0, 0, 0])

        for j in range(self.indices.shape[0]/3):
            vi1 = self.indices[j * 3 + 0]
            vi2 = self.indices[j * 3 + 1]
            vi3 = self.indices[j * 3 + 2]
            
            V1 = self.x[vi1]
            V2 = self.x[vi2]
            V3 = self.x[vi3]

            Vv1 = self.v[vi1]
            Vv2 = self.v[vi2]
            Vv3 = self.v[vi3]

            n = (V2 - V1).cross(V3 - V1)

            A =  n.norm() / 2
            FL = self.KL * A * ti.Vector([0, 0, 1]) 

            n = n / n.norm()
            v = (Vv1 + Vv2 + Vv3)/3 + wind * DT
            if n.dot(v) < 0:
                n = -n

            #FD = self.KD * A * v.dot(n) * (-v)
            FD = 0.5 * self.KD * v.norm()**2 * A * v.dot(n) * (-v)

            if self.WEIGHT[vi1] > 0:
                self.dv[vi1] += DT *(FL + FD) / self.WEIGHT[vi1]
            if self.WEIGHT[vi2] > 0:
                self.dv[vi2] += DT *(FL + FD) / self.WEIGHT[vi2]
            if self.WEIGHT[vi3] > 0:
                self.dv[vi3] += DT * (FL + FD) / self.WEIGHT[vi3]

        for i in range(self.V):
            self.dv[i] /= self.nt[i]
            self.dv[i].z += DT * G
            self.v[i] += self.DAMPING * self.dv[i]

    @ti.kernel
    def make_predictions(self, DT : ti.f32):
        for i in range(self.V):
            self.p[i] = self.x[i] + DT * self.v[i] 


    @ti.kernel
    def apply_correction(self, DT : ti.f32):
        for i in range(self.V):
            self.v[i] = (self.x[i] - self.p[i]) / DT
            self.x[i] = self.p[i]
            if self.fixed[i] == 1:
                self.x[i] = self.POSITIONS[i]

    def fix_point(self, v):
        self.fixed[v] = 1
        self.WEIGHT[v] = 0


    @ti.kernel
    def solve_stretching_constraint(self, ITERATIONS : ti.int32):
        KS = self.KS**ITERATIONS
        for i in range(self.edges.shape[0]/2):
            p1 = self.edges[2 * i]
            p2 = self.edges[2 * i+1]
            
            d = (self.POSITIONS[p1] - self.POSITIONS[p2]).norm()
            l = (self.p[p1] - self.p[p2]).norm()
            n = (self.p[p1] - self.p[p2]) / l

            lagrange = (l- d) / 2

            m = self.WEIGHT[p1] + self.WEIGHT[p2] 

            ti.atomic_add(self.p[p1], -KS * (self.WEIGHT[p1]/m) * lagrange * n)
            ti.atomic_add(self.p[p2], KS * (self.WEIGHT[p2]/m) * lagrange * n)

    @ti.func
    def fit(self, a):
        return min(max(a, -1), 1)

    @ti.kernel
    def solve_bending_constraints(self, ITERATIONS : ti.int32):
        KB = self.KB**ITERATIONS

        for i in range(self.tripairs.shape[0]/4):
            p1 = self.tripairs[4 * i]
            p2 = self.tripairs[4 * i+1]
            p3 = self.tripairs[4 * i+2]
            p4 = self.tripairs[4 * i+3]
    
            v2, w2 = self.p[p2] - self.p[p1], self.POSITIONS[p2] - self.POSITIONS[p1]
            v3, w3 = self.p[p3] - self.p[p1], self.POSITIONS[p3] - self.POSITIONS[p1]
            v4, w4 = self.p[p4] - self.p[p1], self.POSITIONS[p4] - self.POSITIONS[p1]

            vn1, wn1 = v2.cross(v3), w2.cross(w3)
            vn1, wn1 = vn1/vn1.norm(), wn1/wn1.norm()

            vn2, wn2 = v2.cross(v4), w2.cross(w4)
            vn2, wn2 = vn2/vn2.norm(), wn2/wn2.norm()

            vd, wd = vn1.dot(vn2), wn1.dot(wn2)

            vd = self.fit(vd)
            wd = self.fit(wd)

            q3 = (v2.cross(vn2) + vn1.cross(v2) * vd) / (v2.cross(v3)).norm() 
            q4 = (v2.cross(vn1) + vn2.cross(v2) * vd) / (v2.cross(v4)).norm()
            q2 = (-(v3.cross(vn2) + vn1.cross(v3) * vd) / (v2.cross(v3)).norm()) - (v4.cross(vn1) + vn2.cross(v4) * vd) / (v2.cross(v4)).norm()
            q1 = -q2 - q3 - q4

            w = ti.acos(wd)

            sd = -ti.sqrt(1 - vd*vd) * (ti.acos(vd) - w) 

            S = self.WEIGHT[p1] * q1.norm()**2 + self.WEIGHT[p2] * q2.norm()**2 + self.WEIGHT[p3] * q3.norm()**2 + self.WEIGHT[p4] * q4.norm()**2
                    
            if not sd == 0:
                ti.atomic_add(self.p[p1], KB * self.WEIGHT[p1] * (sd/S * q1))
                ti.atomic_add(self.p[p2], KB * self.WEIGHT[p2] * (sd/S * q2))
                ti.atomic_add(self.p[p3], KB * self.WEIGHT[p3] * (sd/S * q3))
                ti.atomic_add(self.p[p4], KB * self.WEIGHT[p4] * (sd/S * q4))


    @ti.kernel
    def solve_collision_constraints(self,  obj : ti.template(), ITERATIONS : ti.int32):
        KC = 1
        for i in range(self.V):
            p = self.p[i]
            x = self.x[i]

            corr = obj.solve_collision_constraint(p, x)
            ti.atomic_add(self.p[i], KC * corr)

    @ti.func
    def triangle_collision(self, p, x, V1, V2, V3):
        E1 = V2 - V1
        E2 = V3 - V1
        D = p - x
        T = x - V1

        P = D.cross(E2)
        Q = T.cross(E1)

        S = 1 / (P.dot(E1))

        t = S * Q.dot(E2)

        u = S * P.dot(T)
        v = S * Q.dot(D)

        if not (u >= 0 and v >= 0 and v + u <= 1):
            t = -1 
            
        return t


    @ti.kernel
    def solve_self_collision_constraints(self, dt : ti.f32):
        for i in range(self.V):
            p = self.p[i]
            x = self.x[i]

            for j in range(self.indices.shape[0]/3):
                vi1 = self.indices[j * 3 + 0]
                vi2 = self.indices[j * 3 + 1]
                vi3 = self.indices[j * 3 + 2]

                V1 = self.p[vi1]
                V2 = self.p[vi2]
                V3 = self.p[vi3]

                V1o = self.x[vi1]
                V2o = self.x[vi2]
                V3o = self.x[vi3]

                
                if not(i == vi1 or i == vi2 or i == vi3):
                    tc = (V1 + V2 + V3) / 3
                    D = (V2 - V1).norm() + (V3 - V1).norm()
                    if (p - tc).norm() < D/2:
                        thickness = 0.002

                        t = self.triangle_collision(p, x, V1, V2, V3)
                        t2 = self.triangle_collision(p, x, V1o, V2o, V3o)
                        
                        if t >= 0 and t <= 1 and t2 >= 0:
                            n = (V2 - V1).cross(V3 - V1)
                            n = n / n.norm()
                            
                            v = p - x
                            if n.dot(v) > 0:
                                n = -n

                            C = n.dot(p - V1) - 2 * thickness
                            M = -1. * n.outer_product(n)
                            M[0, 0] += 1
                            M[1, 1] += 1 
                            M[2, 2] += 1 

                            M = M / n.norm()

                            cp = n
                            c1 = (V2 - V3).cross(M @ n) - n
                            c2 = (V3 - V1).cross(M @ n)
                            c3 = (V2 - V1).cross(M @ n)

                            S = self.WEIGHT[i] * cp.norm()**2 + self.WEIGHT[vi1] * c1.norm()**2 + self.WEIGHT[vi2] * c2.norm()**2 + self.WEIGHT[vi3] * c3.norm()**2

                            
                            ti.atomic_add(self.p[i], -C/S * self.WEIGHT[i] * cp)
                            ti.atomic_add(self.p[vi1], -C/S * self.WEIGHT[vi1] * c1)
                            ti.atomic_add(self.p[vi2], -C/S * self.WEIGHT[vi2] * c2)
                            ti.atomic_add(self.p[vi3], -C/S * self.WEIGHT[vi3] * c3)


                        """
                        thickness = 0.001
                        t = self.triangle_collision(point, old_point, V1, V2, V3)
                        d = point - old_point

                        triangle_normal = (V2 - V1).cross(V3 -V1)
                        triangle_normal = triangle_normal/triangle_normal.norm()

                        if d.dot(triangle_normal) < 0:
                            triangle_normal = -triangle_normal
                            
                        collision_point = old_point + t * d

                        #lagrange = (point - collision_point).dot(triangle_normal) + thickness
                        #lagrange = (point - collision_point).norm() + thickness

                        #corr = (collision_point - p) + thickness * 

                        if t >= 0 and t <= 1:
                            #ti.atomic_add(self.p[i],  - lagrange * triangle_normal)
                            ti.atomic_add(self.p[i], (collision_point - point) - thickness * triangle_normal)
                        """ 

                