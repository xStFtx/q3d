import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

# Quaternion utility functions
def quaternion_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])

def quaternion_to_matrix(q):
    w, x, y, z = q
    n = np.dot(q, q)
    if n < np.finfo(float).eps:
        return np.identity(4)
    q = q / np.sqrt(n)
    w, x, y, z = q
    return np.array([
        [1-2*(y**2+z**2), 2*(x*y-z*w),   2*(x*z+y*w),   0],
        [2*(x*y+z*w),   1-2*(x**2+z**2), 2*(y*z-x*w),   0],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x**2+y**2), 0],
        [0,             0,             0,               1]
    ], dtype=np.float32)

def axis_angle_to_quaternion(axis, angle):
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    s = np.sin(angle/2)
    return np.array([np.cos(angle/2), *(axis * s)])

def quaternion_to_euler(q):
    w, x, y, z = q
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.pi/2 * np.sign(sinp)
    else:
        pitch = np.arcsin(sinp)
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.degrees([roll, pitch, yaw])

def slerp(q1, q2, t):
    """Spherical linear interpolation between two quaternions."""
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = np.dot(q1, q2)
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    q2_ortho = q2 - q1 * dot
    q2_ortho /= np.linalg.norm(q2_ortho)
    return q1 * np.cos(theta) + q2_ortho * np.sin(theta)

# --- OpenGL Text Rendering Helper ---
def draw_text_2d(text, x, y, font_size=18, color=(1,1,1)):
    # Simple OpenGL bitmap text using GLUT (if available)
    try:
        from OpenGL.GLUT import glutInit, glutBitmapCharacter, GLUT_BITMAP_HELVETICA_18
        glutInit()
        glColor3f(*color)
        glWindowPos2f(x, y)
        for ch in text:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(ch))
    except Exception:
        pass  # GLUT not available, skip text
# --- End OpenGL Text Rendering Helper ---

# Cube vertices and edges
vertices = [
    [1, 1, -1],
    [1, -1, -1],
    [-1, -1, -1],
    [-1, 1, -1],
    [1, 1, 1],
    [1, -1, 1],
    [-1, -1, 1],
    [-1, 1, 1]
]
edges = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7)
]

def draw_cube():
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()

def draw_axes():
    glBegin(GL_LINES)
    # X axis (red)
    glColor3f(1, 0, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(2, 0, 0)
    # Y axis (green)
    glColor3f(0, 1, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 2, 0)
    # Z axis (blue)
    glColor3f(0, 0, 1)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, 2)
    glEnd()
    glColor3f(1, 1, 1)  # Reset color

def draw_colored_cube():
    faces = [
        (0, 1, 2, 3),  # back
        (4, 5, 6, 7),  # front
        (0, 4, 5, 1),  # right
        (3, 7, 6, 2),  # left
        (0, 4, 7, 3),  # top
        (1, 5, 6, 2),  # bottom
    ]
    colors = [
        (1, 0, 0),  # red
        (0, 1, 0),  # green
        (0, 0, 1),  # blue
        (1, 1, 0),  # yellow
        (1, 0, 1),  # magenta
        (0, 1, 1),  # cyan
    ]
    glBegin(GL_QUADS)
    for i, face in enumerate(faces):
        glColor3fv(colors[i])
        for vertex in face:
            glVertex3fv(vertices[vertex])
    glEnd()
    glColor3f(1, 1, 1)  # Reset color

def draw_text(screen, text, pos, color=(255,255,255)):
    font = pygame.font.SysFont('consolas', 20)
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, pos)

def draw_grid(size=10, step=1):
    glColor3f(0.5, 0.5, 0.5)
    glBegin(GL_LINES)
    for i in range(-size, size+1, step):
        glVertex3f(i, 0, -size)
        glVertex3f(i, 0, size)
        glVertex3f(-size, 0, i)
        glVertex3f(size, 0, i)
    glEnd()
    glColor3f(1, 1, 1)

def draw_sphere(radius=1, slices=24, stacks=16):
    for i in range(stacks):
        lat0 = np.pi * (-0.5 + float(i) / stacks)
        z0 = np.sin(lat0)
        zr0 = np.cos(lat0)
        lat1 = np.pi * (-0.5 + float(i+1) / stacks)
        z1 = np.sin(lat1)
        zr1 = np.cos(lat1)
        glBegin(GL_QUAD_STRIP)
        for j in range(slices+1):
            lng = 2 * np.pi * float(j) / slices
            x = np.cos(lng)
            y = np.sin(lng)
            glColor3f(0.2 + 0.8 * abs(x), 0.2 + 0.8 * abs(y), 0.7)
            glVertex3f(radius * x * zr0, radius * y * zr0, radius * z0)
            glVertex3f(radius * x * zr1, radius * y * zr1, radius * z1)
        glEnd()
    glColor3f(1, 1, 1)

def save_scene(objects, filename='scene.json'):
    import json
    data = []
    for o in objects:
        data.append({
            'shape': o.shape,
            'orientation': o.orientation.tolist(),
            'scale': o.scale,
            'translate': o.translate.tolist(),
            'trail': [q.tolist() for q in o.trail],
            'keyframes': [q.tolist() for q in o.keyframes],
        })
    with open(filename, 'w') as f:
        json.dump(data, f)
    print(f"Scene saved to {filename}")

def load_scene(filename='scene.json'):
    import json
    objects = []
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        for d in data:
            o = Object3D(d['shape'])
            o.orientation = np.array(d['orientation'], dtype=np.float32)
            o.scale = d['scale']
            o.translate = np.array(d['translate'], dtype=np.float32)
            o.trail = [np.array(q, dtype=np.float32) for q in d.get('trail', [])]
            o.keyframes = [np.array(q, dtype=np.float32) for q in d.get('keyframes', [])]
            objects.append(o)
        print(f"Scene loaded from {filename}")
    except Exception as e:
        print(f"Failed to load scene: {e}")
    return objects

class Camera:
    def __init__(self):
        self.distance = 7.0
        self.azimuth = 0.0
        self.elevation = 0.0
        self.pan = np.array([0.0, 0.0])
        self.last_mouse = None
        self.mode = None  # 'rotate', 'pan'
    def apply(self):
        glTranslatef(self.pan[0], self.pan[1], -self.distance)
        glRotatef(self.elevation, 1, 0, 0)
        glRotatef(self.azimuth, 0, 1, 0)

class Object3D:
    def __init__(self, shape='cube'):
        self.orientation = np.array([1,0,0,0], dtype=np.float32)
        self.target_orientation = axis_angle_to_quaternion([0, 1, 0], np.pi/2)
        self.animating = False
        self.anim_t = 0.0
        self.anim_speed = 0.01
        self.scale = 1.0
        self.translate = np.array([0.0, 0.0, 0.0])
        self.shape = shape
        self.keyframes = []
        self.playing = False
        self.keyframe_idx = 0
        self.trail = []
    def draw(self):
        glPushMatrix()
        glMultMatrixf(quaternion_to_matrix(self.orientation).T)
        glScalef(self.scale, self.scale, self.scale)
        glTranslatef(*self.translate)
        draw_axes()
        if self.shape == 'cube':
            glColor3f(1, 1, 1)
            draw_colored_cube()
        elif self.shape == 'sphere':
            draw_sphere(radius=1.2)
        glPopMatrix()
    def update(self):
        if self.animating:
            self.anim_t += self.anim_speed
            if self.anim_t >= 1.0:
                self.anim_t = 1.0
                self.animating = False
            self.orientation = slerp(self.orientation, self.target_orientation, self.anim_t)
        if self.playing and self.keyframes:
            self.orientation = self.keyframes[self.keyframe_idx]
            self.keyframe_idx += 1
            if self.keyframe_idx >= len(self.keyframes):
                self.keyframe_idx = 0
                self.playing = False
        # Add orientation to trail
        if len(self.trail) == 0 or not np.allclose(self.orientation, self.trail[-1]):
            self.trail.append(self.orientation.copy())
        if len(self.trail) > 200:
            self.trail.pop(0)
    def draw_trail(self):
        if len(self.trail) < 2:
            return
        glColor3f(1, 0.5, 0.2)
        glBegin(GL_LINE_STRIP)
        for q in self.trail:
            # Map quaternion orientation to a 3D point (e.g., rotate a unit vector)
            v = np.array([0, 0, 2, 0])  # Homogeneous for mat mult
            m = quaternion_to_matrix(q)
            p = m @ v
            glVertex3f(p[0], p[1], p[2])
        glEnd()
        glColor3f(1, 1, 1)

def main():
    pygame.init()
    display = (800,600)
    screen = pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    glTranslatef(0.0,0.0, -5)

    glClearColor(0.2, 0.2, 0.2, 1.0)  # Set a visible gray background
    glEnable(GL_DEPTH_TEST)  # Enable depth testing for correct 3D rendering

    objects = [Object3D('cube')]
    current_obj = 0
    camera = Camera()
    print("Controls: TAB=switch shape, N=new object, M=next object, K=record keyframe, P=play keyframes, 1/2/3=preset spins, R=reset, SPACE=animate, T=random target, WASDZX=move, +/-=scale, Arrow/QE=rotate")

    help_lines = [
        "Controls:",
        "TAB: Switch shape (cube/sphere)",
        "N: New object, M: Next object", 
        "K: Record keyframe, P: Play keyframes", 
        "1/2/3: Preset spins (X/Y/Z)",
        "R: Reset, SPACE: Animate, T: Random target",
        "WASDZX: Move, +/-: Scale, Arrow/QE: Rotate",
        "S: Save scene, L: Load scene",
        "E: Edit quaternion/Euler (toggle mode)",
        "D: Duplicate object, DEL: Delete object",
        "F: Plot quaternion function (toggle)",
        "ESC: Quit"
    ]
    edit_mode = None  # None, 'quat', 'euler'
    edit_idx = 0
    edit_val = 0.0
    plot_func = False

    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    camera.mode = 'rotate'
                    camera.last_mouse = pygame.mouse.get_pos()
                elif event.button == 3:
                    camera.mode = 'pan'
                    camera.last_mouse = pygame.mouse.get_pos()
                elif event.button == 4:  # Scroll up
                    camera.distance -= 0.5
                elif event.button == 5:  # Scroll down
                    camera.distance += 0.5
            elif event.type == pygame.MOUSEBUTTONUP:
                camera.mode = None
            elif event.type == pygame.MOUSEMOTION and camera.mode:
                x, y = pygame.mouse.get_pos()
                dx, dy = x - camera.last_mouse[0], y - camera.last_mouse[1]
                if camera.mode == 'rotate':
                    camera.azimuth += dx * 0.5
                    camera.elevation += dy * 0.5
                elif camera.mode == 'pan':
                    camera.pan[0] += dx * 0.01
                    camera.pan[1] -= dy * 0.01
                camera.last_mouse = (x, y)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_TAB:
                    if objects[current_obj].shape == 'cube':
                        objects[current_obj].shape = 'sphere'
                    else:
                        objects[current_obj].shape = 'cube'
                if event.key == pygame.K_n:
                    objects.append(Object3D('cube'))
                    current_obj = len(objects)-1
                if event.key == pygame.K_m:
                    current_obj = (current_obj + 1) % len(objects)
                if event.key == pygame.K_r:
                    objects[current_obj].orientation = np.array([1,0,0,0], dtype=np.float32)
                    objects[current_obj].scale = 1.0
                    objects[current_obj].translate = np.array([0.0, 0.0, 0.0])
                if event.key == pygame.K_SPACE:
                    obj = objects[current_obj]
                    obj.animating = True
                    obj.anim_t = 0.0
                if event.key == pygame.K_t:
                    axis = np.random.randn(3)
                    axis /= np.linalg.norm(axis)
                    angle = np.random.uniform(0, 2*np.pi)
                    objects[current_obj].target_orientation = axis_angle_to_quaternion(axis, angle)
                if event.key == pygame.K_k:
                    # Record keyframe
                    obj = objects[current_obj]
                    obj.keyframes.append(obj.orientation.copy())
                    print(f"Keyframe recorded for object {current_obj} (total: {len(obj.keyframes)})")
                if event.key == pygame.K_p:
                    obj = objects[current_obj]
                    if obj.keyframes:
                        obj.playing = True
                        obj.keyframe_idx = 0
                        print(f"Playing keyframes for object {current_obj}")
                if event.key == pygame.K_1:
                    # Preset spin X
                    obj = objects[current_obj]
                    obj.target_orientation = axis_angle_to_quaternion([1,0,0], np.pi)
                    obj.animating = True
                    obj.anim_t = 0.0
                if event.key == pygame.K_2:
                    # Preset spin Y
                    obj = objects[current_obj]
                    obj.target_orientation = axis_angle_to_quaternion([0,1,0], np.pi)
                    obj.animating = True
                    obj.anim_t = 0.0
                if event.key == pygame.K_3:
                    # Preset spin Z
                    obj = objects[current_obj]
                    obj.target_orientation = axis_angle_to_quaternion([0,0,1], np.pi)
                    obj.animating = True
                    obj.anim_t = 0.0
                if event.key == pygame.K_s:
                    save_scene(objects)
                if event.key == pygame.K_l:
                    loaded = load_scene()
                    if loaded:
                        objects.clear()
                        objects.extend(loaded)
                        current_obj = 0
                if event.key == pygame.K_e:
                    if edit_mode is None:
                        edit_mode = 'quat'
                        edit_idx = 0
                    elif edit_mode == 'quat':
                        edit_mode = 'euler'
                        edit_idx = 0
                    else:
                        edit_mode = None
                if event.key == pygame.K_d:
                    # Duplicate current object
                    import copy
                    new_obj = copy.deepcopy(objects[current_obj])
                    objects.append(new_obj)
                    current_obj = len(objects)-1
                if event.key == pygame.K_DELETE:
                    if len(objects) > 1:
                        del objects[current_obj]
                        current_obj = max(0, current_obj-1)
                if event.key == pygame.K_f:
                    plot_func = not plot_func
                if edit_mode:
                    if event.key == pygame.K_RIGHT:
                        edit_idx = (edit_idx + 1) % 4 if edit_mode == 'quat' else (edit_idx + 1) % 3
                    if event.key == pygame.K_LEFT:
                        edit_idx = (edit_idx - 1) % 4 if edit_mode == 'quat' else (edit_idx - 1) % 3
                    if event.key == pygame.K_UP:
                        if edit_mode == 'quat':
                            obj.orientation[edit_idx] += 0.05
                        else:
                            euler = quaternion_to_euler(obj.orientation)
                            euler[edit_idx] += 5
                            # Convert back to quaternion
                            from scipy.spatial.transform import Rotation as R
                            q = R.from_euler('xyz', euler, degrees=True).as_quat()
                            obj.orientation = np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)  # w,x,y,z
                    if event.key == pygame.K_DOWN:
                        if edit_mode == 'quat':
                            obj.orientation[edit_idx] -= 0.05
                        else:
                            euler = quaternion_to_euler(obj.orientation)
                            euler[edit_idx] -= 5
                            from scipy.spatial.transform import Rotation as R
                            q = R.from_euler('xyz', euler, degrees=True).as_quat()
                            obj.orientation = np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return

        keys = pygame.key.get_pressed()
        obj = objects[current_obj]
        # Control rotation with arrow keys
        if keys[pygame.K_LEFT]:
            q = axis_angle_to_quaternion([0,1,0], np.radians(2))
            obj.orientation = quaternion_mult(q, obj.orientation)
        if keys[pygame.K_RIGHT]:
            q = axis_angle_to_quaternion([0,1,0], -np.radians(2))
            obj.orientation = quaternion_mult(q, obj.orientation)
        if keys[pygame.K_UP]:
            q = axis_angle_to_quaternion([1,0,0], np.radians(2))
            obj.orientation = quaternion_mult(q, obj.orientation)
        if keys[pygame.K_DOWN]:
            q = axis_angle_to_quaternion([1,0,0], -np.radians(2))
            obj.orientation = quaternion_mult(q, obj.orientation)
        if keys[pygame.K_q]:
            q = axis_angle_to_quaternion([0,0,1], np.radians(2))
            obj.orientation = quaternion_mult(q, obj.orientation)
        if keys[pygame.K_e]:
            q = axis_angle_to_quaternion([0,0,1], -np.radians(2))
            obj.orientation = quaternion_mult(q, obj.orientation)
        if keys[pygame.K_w]:
            obj.translate[1] += 0.05
        if keys[pygame.K_s]:
            obj.translate[1] -= 0.05
        if keys[pygame.K_a]:
            obj.translate[0] -= 0.05
        if keys[pygame.K_d]:
            obj.translate[0] += 0.05
        if keys[pygame.K_z]:
            obj.translate[2] += 0.05
        if keys[pygame.K_x]:
            obj.translate[2] -= 0.05
        if keys[pygame.K_EQUALS] or keys[pygame.K_KP_PLUS]:
            obj.scale *= 1.02
        if keys[pygame.K_MINUS] or keys[pygame.K_KP_MINUS]:
            obj.scale /= 1.02

        # Update all objects
        for o in objects:
            o.update()

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        camera.apply()
        draw_grid()
        for o in objects:
            o.draw_trail()
        for o in objects:
            o.draw()
        # Quaternion function plotter
        if plot_func:
            glColor3f(0.2, 1, 0.2)
            glBegin(GL_LINE_STRIP)
            for t in np.linspace(0, 1, 100):
                # Example: interpolate between identity and current orientation
                q = slerp(np.array([1,0,0,0], dtype=np.float32), obj.orientation, t)
                v = np.array([0, 1, 0, 0])
                m = quaternion_to_matrix(q)
                p = m @ v
                glVertex3f(p[0], p[1], p[2])
            glEnd()
            glColor3f(1, 1, 1)

        # --- OpenGL text overlay ---
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, display[0], 0, display[1], -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        y = display[1] - 20
        for line in help_lines:
            draw_text_2d(line, 10, y)
            y -= 20
        # Show current quaternion/Euler
        euler_angles = quaternion_to_euler(obj.orientation)
        q_str = f"Q: [{obj.orientation[0]:.2f}, {obj.orientation[1]:.2f}, {obj.orientation[2]:.2f}, {obj.orientation[3]:.2f}]"
        euler_str = f"Euler: Roll={euler_angles[0]:.1f} Pitch={euler_angles[1]:.1f} Yaw={euler_angles[2]:.1f}"
        draw_text_2d(f"Object {current_obj} | {q_str}", 10, 40)
        draw_text_2d(euler_str, 10, 20)
        if edit_mode:
            draw_text_2d(f"EDIT MODE: {'QUAT' if edit_mode=='quat' else 'EULER'} idx={edit_idx}", 300, 40, color=(1,0.8,0.2))
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        # --- End OpenGL text overlay ---

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
