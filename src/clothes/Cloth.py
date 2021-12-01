import taichi as ti
from abc import ABCMeta, abstractmethod
from src.objects.Object import Object

@ti.data_oriented
class Cloth(object, metaclass=ABCMeta):
    def __init__(self, V, POSITIONS, VELOCITIES, indices, edges, tripairs, color, KS, KC, KB, DAMPING):
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
        self.DAMPING = DAMPING
        self.objs = None

        self.vertices = ti.Vector.field(3, float, self.V)
        self.x = ti.Vector.field(3, float, self.V)
        self.v = ti.Vector.field(3, float, self.V)
        self.p = ti.Vector.field(3, float, self.V)
        self.x_delta = ti.Vector.field(3, float, self.V)

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
    def external_forces(self, G : ti.f32, DT : ti.f32):
        for i in range(self.V):
            self.v[i].z = DT * G
            self.v[i] *= self.DAMPING 

    @ti.kernel
    def make_predictions(self, DT : ti.f32):
        for i in range(self.V):
            self.p[i] = self.x[i] + DT * self.v[i] 

    @ti.kernel
    def update_predictions(self):
        for i in range(self.V):
            self.p[i] += self.x_delta[i]
            self.x_delta[i] = ti.Vector([0, 0, 0])


    @ti.kernel
    def apply_correction(self, DT : ti.f32):
        for i in range(self.V):
            self.p[i] += self.x_delta[i]
            self.v[i] = (self.x[i] - self.p[i]) / DT
            self.x[i] = self.p[i]

            self.x_delta[i] = ti.Vector([0, 0, 0])


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

            #self.x_delta[p1] -= self.KS * lagrange * n
            #self.x_delta[p2] += self.KS * lagrange * n

            ti.atomic_add(self.x_delta[p1], -KS * lagrange * n)
            ti.atomic_add(self.x_delta[p2], KS * lagrange * n)

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

            S = q1.norm()**2 + q2.norm()**2 + q3.norm()**2 + q4.norm()**2
                    
            if not sd == 0:
                ti.atomic_add(self.x_delta[p1], KB * (sd/S * q1))
                ti.atomic_add(self.x_delta[p2], KB * (sd/S * q2))
                ti.atomic_add(self.x_delta[p3], KB * (sd/S * q3))
                ti.atomic_add(self.x_delta[p4], KB * (sd/S * q4))

    """
    def collision_kernel(self, ITERATIONS : ti.int32, idx: ti.int32):
        
        KC = self.KC**ITERATIONS
        for i in range(self.V):
            if self.objs[idx].collides(self.p[i]):
                correction_term = - KC * self.objs[int(idx)].solve_collision_constraint(self.p[i])
                ti.atomic_add(self.x_delta[i], correction_term)
    """


    def solve_collision_constraints(self, ITERATIONS, obj):
        data = ti.Vector.field(3, float, self.V)
        result = ti.Vector.field(3, float, self.V)

        for i in range(self.V):
            data[i] = self.p[i]

        o.set_data(data, result)
        o.solve_collision_constraint_for_all()

        
        