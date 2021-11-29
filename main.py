import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

# ------- PARAMETERS ---------

# Solver Properties
GRAVITY = -10               # gravity
DAMPING = 0.3              # 0-1 where 1 = no damping
SOLVER_ITERATIONS = 2      # Iterations for solver optimization
DT = 0.005                 # Time delta

# Material Properties
N = 128                  # NxN grid
CLOTH_SIZE = 0.5            # Size in 3d coords
DELTA_SIZE = CLOTH_SIZE / N # space between 2 grid points

STRETCHING_STIFFNESS = 0.6 # 0-1 where 0 = stiff
WD = np.pi                  # initial angle between triangles
BENDING_STIFFNESS = 0.4   # 0-1 where 1 = no bending
COLLISION_STIFFNESS = 0.4  # 0-1 where 1 = max repulsion

# Cube Properties
cube_corner = ti.Vector([-0.2, -0.2, -0.4])
cube_width = 0.4
cube_height = 0.4
cube_depth = 0.2

# sphere properties
sphere_radius = 0.4
sphere_center = ti.Vector.field(3, float, (1, ))
sphere_center[0] = ti.Vector([0, 0, -sphere_radius])

#Window Properties
RESX, RESY = 1600, 900      # window resolution

p = ti.Vector.field(3, float, (N, N)) # cloth particle predictions
x = ti.Vector.field(3, float, (N, N)) # cloth particle positions
v = ti.Vector.field(3, float, (N, N)) # cloth particle velocities

num_triangles = (N - 1) * (N - 1) * 2
indices = ti.field(int, num_triangles * 3)
vertices = ti.Vector.field(3, float, N * N)



def init_cloth():
    for i, j in ti.ndrange(N, N):
        x[i, j] = ti.Vector([
            i * DELTA_SIZE-CLOTH_SIZE/2, 
            j * DELTA_SIZE-CLOTH_SIZE/2,
            0.010001
        ])
        v[i, j] = ti.Vector([0, 0, 0])

@ti.kernel
def init_triangles():
    for i, j in ti.ndrange(N, N):
        if i < N - 1 and j < N - 1:
            square_id = (i * (N - 1)) + j
            # 1st triangle of the square
            indices[square_id * 6 + 0] = i * N + j
            indices[square_id * 6 + 1] = (i + 1) * N + j
            indices[square_id * 6 + 2] = i * N + (j + 1)
            # 2nd triangle of the square
            indices[square_id * 6 + 3] = (i + 1) * N + j + 1
            indices[square_id * 6 + 4] = i * N + (j + 1)
            indices[square_id * 6 + 5] = (i + 1) * N + j

@ti.kernel
def set_vertices():
    for i, j in ti.ndrange(N, N):
        vertices[i * N + j] = x[i, j]

init_cloth()
init_triangles()
set_vertices()


window = ti.ui.Window("Cloth Simulation (PBD)", (RESX, RESY), vsync=True)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()

x_delta = ti.Vector.field(3, float, (N, N))

@ti.func
def solve_stretching(stride):
    KS = (1 - STRETCHING_STIFFNESS)**SOLVER_ITERATIONS # stretching coeff

    # stretching constraint
    for i, j in ti.ndrange(N, N):
        # right
        if j + stride < N:
            d = stride * DELTA_SIZE
            n = (p[i, j] - p[i,j+stride]) / (p[i, j] - p[i,j+stride]).norm()
            lagrange = ((p[i, j] - p[i,j+stride]).norm() - d) / 2

            x_delta[i, j]   += -KS * lagrange * n
            x_delta[i, j+stride] += KS * lagrange * n
        
        # diagonal (down right)
        if j + stride < N and i + stride < N:
            d =  ti.sqrt(2*((DELTA_SIZE * stride)**2))
            n = (p[i, j] - p[i+stride,j+stride]) / (p[i, j] - p[i+stride,j+stride]).norm()
            lagrange = ((p[i, j] - p[i+stride,j+stride]).norm() - d) / 2

            x_delta[i, j]       += - KS * lagrange * n
            x_delta[i+stride, j+stride]   +=  KS * lagrange * n
        
        # down
        if i + stride < N:
            d = stride * DELTA_SIZE
            n = (p[i, j] - p[i+stride,j]) / (p[i, j] - p[i+stride,j]).norm()
            lagrange = ((p[i, j] - p[i+stride,j]).norm() - d) / 2

            x_delta[i, j]   += -KS * lagrange * n
            x_delta[i+stride, j] += KS * lagrange * n
        
        # up
        if i - stride >= 0:
            d = stride * DELTA_SIZE
            n = (p[i, j] - p[i-stride,j]) / (p[i, j] - p[i-stride,j]).norm()
            lagrange = ((p[i, j] - p[i-stride,j]).norm() - d) / 2

            x_delta[i, j]   += - KS * lagrange * n
            x_delta[i-stride,j] +=  KS * lagrange * n
        
        # left
        if j - stride >= 0:
            d = stride * DELTA_SIZE
            n = (p[i, j] - p[i,j-stride]) / (p[i, j] - p[i,j-stride]).norm()
            lagrange = ((p[i, j] - p[i,j-stride]).norm() - d) / 2

            x_delta[i, j]   += -KS * lagrange * n
            x_delta[i, j-stride] += KS *   lagrange * n
        
        # diagonal (up left)
        if j - stride >= 0 and i - stride >= 0:
            d = stride * ti.sqrt(2*(DELTA_SIZE**2))
            n = (p[i, j] - p[i-stride,j-stride]) / (p[i, j] - p[i-stride,j-stride]).norm()
            lagrange = ((p[i, j] - p[i-stride,j-stride]).norm() - d) / 2

            x_delta[i, j]       += -KS * lagrange * n
            x_delta[i-stride, j-stride]   += KS *  lagrange * n

        # diagonal (up right)
        if j + stride < N and i - stride >= 0:
            d = stride * ti.sqrt(2*(DELTA_SIZE**2))
            n = (p[i, j] - p[i-stride,j+stride]) / (p[i, j] - p[i-stride,j+stride]).norm()
            lagrange = ((p[i, j] - p[i-stride,j+stride]).norm() - d) / 2

            x_delta[i, j]       += -KS * lagrange * n
            x_delta[i-stride, j+stride]   += KS *  lagrange * n
        
        # diagonal (down left)
        if j - stride >= 0 and i + stride < N:
            d = stride * ti.sqrt(2*(DELTA_SIZE**2))
            n = (p[i, j] - p[i+stride,j-stride]) / (p[i, j] - p[i+stride,j-stride]).norm()
            lagrange = ((p[i, j] - p[i+stride,j-stride]).norm() - d) / 2

            x_delta[i, j]       += -KS * lagrange * n
            x_delta[i+stride, j-stride]   += KS *  lagrange * n
        
@ti.func
def isnan(x):
    return not (x < 0 or 0 < x or x == 0)

@ti.func
def get_bending_correction(p1, p2, p3, p4):
    op2 = p2 - p1
    op3 = p3 - p1
    op4 = p4 - p1

    n1 = op2.cross(op3)
    n1 = n1 / n1.norm()

    n2 = op2.cross(op4)
    n2 = n2 / n2.norm()

    d = n1.dot(n2)

    while d < -1:
        d += 2
    while d > 1:
        d -= 2

    q3 = (op2.cross(n2) + n1.cross(op2) * d) / (op2.cross(op3)).norm() 
    q4 = (op2.cross(n1) + n2.cross(op2) * d) / (op2.cross(op4)).norm()
    q2 = (-(op3.cross(n2) + n1.cross(op3) * d) / (op2.cross(op3)).norm()) - (op4.cross(n1) + n2.cross(op4) * d) / (op2.cross(op4)).norm()
    q1 = -q2 - q3 - q4

    

    

    sd = -ti.sqrt(1 - d*d) * (ti.acos(d) - WD) 

    S = q1.norm()**2 + q2.norm()**2 + q3.norm()**2 + q4.norm()**2
    
    delta_p1, delta_p2, delta_p3, delta_p4 = ti.Vector([0, 0, 0]), ti.Vector([0, 0, 0]), ti.Vector([0, 0, 0]), ti.Vector([0, 0, 0])

    if not sd == 0:
        delta_p1 = (sd/S * q1)
        delta_p2 = (sd/S * q2)
        delta_p3 = (sd/S * q3)
        delta_p4 = (sd/S * q4)

    return delta_p1, delta_p2, delta_p3, delta_p4

@ti.func
def solve_bending(stride):
    KB = (BENDING_STIFFNESS)**SOLVER_ITERATIONS


    # diag
    for i, j in ti.ndrange(N, N):
        if i+stride < N and j + stride < N:
            d1, d2, d3, d4 = get_bending_correction(p[i, j], p[i+1, j+1], p[i, j+1], p[i+1,j])
            
            x_delta[i, i] += KB * d1
            x_delta[i+1, j+1] += KB * d2
            x_delta[i, j+1] += KB * d3
            x_delta[i+1, j] += KB * d4
    
    # left
    for i, j in ti.ndrange(N, N):
        if i-stride >= 0 and i+stride < N and j + stride < N:
            d1, d2, d3, d4 = get_bending_correction(p[i, j], p[i, j+1], p[i-1, j], p[i+1,j+1])
            
            x_delta[i, i] += KB * d1
            x_delta[i, j+1] += KB * d2
            x_delta[i-1, j] += KB * d3
            x_delta[i+1, j+1] += KB * d4

    # down
    for i, j in ti.ndrange(N, N):
        if j-stride >= 0 and i+stride < N and j + stride < N:
            d1, d2, d3, d4 = get_bending_correction(p[i, j], p[i+1, j], p[i+1, j+1], p[i,j-1])
            
            x_delta[i, i] += KB * d1
            x_delta[i+1, j] += KB * d2
            x_delta[i+1, j+1] += KB * d3
            x_delta[i, j-1] += KB * d4

@ti.func
def sign(a):
    return 1 if a >= 0 else -1

box_collision_normals = ti.Vector.field(3, float, 3)

@ti.func
def solve_box_collison(corner, width, depth, height):
    DREST = 0.01
    KC = COLLISION_STIFFNESS ** SOLVER_ITERATIONS

    
    for i, j in ti.ndrange(N, N):

        # check collision
        if (p[i, j].x > corner.x - DREST and p[i, j].x < corner.x + width + DREST and 
            p[i, j].y > corner.y - DREST and p[i, j].y < corner.y + depth + DREST and
            p[i, j].z > corner.z - DREST and p[i, j].z < corner.z + height + DREST):
            
            box_center = corner + ti.Vector([width/2, depth/2, height/2])
            
            box_collision_normals[0] = ti.Vector([1, 0, 0])
            box_collision_normals[1] = ti.Vector([0, 1, 0])
            box_collision_normals[2] = ti.Vector([0, 0, 1])
            

            point_vector = p[i,j] - box_center
            point_vector = point_vector / point_vector.norm()
            biggest = 0
            index = -1
            
            for k in range(3):
                kn = box_collision_normals[k]
                if abs(kn.dot(point_vector)) > abs(biggest):
                    
                    index = k
                    biggest = kn.dot(point_vector)
                
            correction_direction = sign(biggest) * box_collision_normals[index]

            eps = 0.2
            cos45 = 1 / ti.sqrt(2) 
            if not (biggest > cos45 - eps and biggest < cos45 + eps):
                x_delta[i,j] += - KC * (correction_direction.dot(p[i,j]) - DREST) * correction_direction

@ti.func
def solve_sphere_collison(center, radius):
    DREST = 0.01
    KC = COLLISION_STIFFNESS ** SOLVER_ITERATIONS

    
    
    for i, j in ti.ndrange(N, N):
        point_vector = p[i,j] - center
        
        if (point_vector.norm() <  DELTA_SIZE + radius + DREST):
            n = point_vector = point_vector / point_vector.norm()
            x_delta[i,j] += - KC * (n.dot(p[i,j]) - DREST) * n
            
            
        """
        if p[i,j].z < 0. + DREST and p[i,j].x < 0.2 and p[i,j].x > -0.2 and p[i,j].y < 0.2 and p[i,j].y > -0.2:
            n = ti.Vector([0, 0, 1.0])
            x_delta[i,j] += - KC * (n.dot(p[i,j]) - DREST) * n
        
        # left
        if p[i,j].x < 0.2 + DREST and p[i,j].x < 0.2 and p[i,j].z > 0:
            n = ti.Vector([-1, 0, 0])
            x_delta[i,j] += - KC * (n.dot(p[i,j]) - DREST) * n

        # right
        if p[i,j].x > -0.2 - DREST and p[i,j].x < -0.2 and p[i,j].z > 0:
            n = ti.Vector([1, 0, 0])
            x_delta[i,j] += - KC * (n.dot(p[i,j]) - DREST) * n
        
        # back
        if p[i,j].y > 0.2 + DREST and p[i,j].z < 0:
            n = ti.Vector([0, -1, 0])
            x_delta[i,j] += - KC * (n.dot(p[i,j]) - DREST) * n

        # front
        if p[i,j].y < -0.2 - DREST and p[i,j].z < 0:
            n = ti.Vector([0, 1, 0])
            x_delta[i,j] += - KC * (n.dot(p[i,j]) - DREST) * n
        """
@ti.func
def solve_constraints(stride):
    # particle corrections
    for i, j in ti.ndrange(N, N):
        x_delta[i, j] = ti.Vector([0, 0, 0])
    
    solve_stretching(stride)
    for i, j in ti.ndrange(N, N):
        p[i, j] += x_delta[i, j]

    for i, j in ti.ndrange(N, N):
        x_delta[i, j] = ti.Vector([0, 0, 0])
    solve_bending(stride)
    for i, j in ti.ndrange(N, N):
        p[i, j] += x_delta[i, j]
        
    for i, j in ti.ndrange(N, N):
        x_delta[i, j] = ti.Vector([0, 0, 0])
    solve_box_collison(cube_corner, cube_width, cube_width, cube_height)
    #solve_sphere_collison(sphere_center[0], sphere_radius)

    for i, j in ti.ndrange(N, N):
        p[i, j] += x_delta[i, j]

         

@ti.kernel
def step():
    # Gravity, external forces
    for i in ti.grouped(v):
        v[i].z = DT * GRAVITY
        v[i] *= DAMPING 

    # make predictions for the positions
    for i in ti.grouped(x):
        p[i] = x[i] + DT * v[i]  

    # make and solve constraints

    for _ in range(1):
        for i in range(SOLVER_ITERATIONS):
            solve_constraints(1)
            #solve_constraints(10)
        
        

    # set obtained positions and update velocities accordingly
    for i in ti.grouped(x):
        v[i] = (x[i] - p[i]) / DT
        x[i] = p[i]


# orientation

origin = ti.Vector.field(3, float, (1, ))
or_x = ti.Vector.field(3, float, (1, ))
or_y = ti.Vector.field(3, float, (1, ))
or_z = ti.Vector.field(3, float, (1, ))

origin[0] = ti.Vector([0, 0, 0])
or_x[0] = ti.Vector([0.1, 0, 0])
or_y[0] = ti.Vector([0, 0.1, 0])
or_z[0] = ti.Vector([0, 0, 0.1])



CUBE_FACES = [
    [0, 1, 2, 3],
    [4, 7, 6, 5],
    [0, 4, 5, 1],
    [3, 2, 6, 7],
    [1, 5, 6, 2],
    [0, 3, 7, 4]
]


def make_tri_from_quad(list):
    return [list[0], list[2], list[1], list[0], list[2], list[3]]
    
def get_cube_indices():
    res = [make_tri_from_quad(x) for x in CUBE_FACES]
    return [item for sublist in res for item in sublist]

def make_cuboid(corner, width, depth, height):
    cuboid = ti.Vector.field(3, float, 8)
    width = ti.Vector([width, 0, 0])
    depth = ti.Vector([0, depth, 0])
    height = ti.Vector([0, 0, height])
    cuboid[0] = corner
    cuboid[1] = cuboid[0] + width
    cuboid[2] = cuboid[1] + height
    cuboid[3] = cuboid[2] - width
    cuboid[4] = cuboid[0] + depth
    cuboid[5] = cuboid[4] + width
    cuboid[6] = cuboid[5] + height
    cuboid[7] = cuboid[6] - width

    indices = ti.field(int, 3 * 12)
    for i, e in enumerate(get_cube_indices()):
        indices[i] = e
    return cuboid, indices


vm = ti.VideoManager(output_dir="./results", framerate=24)

cube_vertices, cube_indices = make_cuboid(cube_corner, cube_width, cube_width, cube_height)   
    #
while window.running:

    #solver here
    step()
    set_vertices()

    camera.position(0.4, -1, 0.3)
    camera.lookat(0, 0, 0)
    camera.up(0, 0, 1)
    scene.set_camera(camera)
    
    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))

    scene.mesh(vertices,
               indices=indices,
               color=(0.8, 0, 0),
               two_sided=True)
    
    #scene.particles(sphere_center, radius=sphere_radius, color=(0.5, 0, 0))

    scene.mesh(cube_vertices,
               indices=cube_indices,
               color=(0, 0, 0.9),
               two_sided=True)
    
    #scene.particles(origin, radius=0.05, color=(1, 1, 1)) # origin
    #scene.particles(or_x, radius=0.05, color=(1, 0, 0)) # x
    #scene.particles(or_y, radius=0.05, color=(0, 1, 0)) # y
    #scene.particles(or_z, radius=0.05, color=(0, 0, 1)) # z
    #scene.particles(ball_center, radius=ball_radius, color=(0.5, 0, 0))
    canvas.scene(scene)
    #vm.write_frame(window.get_image().to_numpy())
    window.show()

vm.make_video(gif=True, mp4=True)