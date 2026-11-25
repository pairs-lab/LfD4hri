"""Deterministic causal ground-truth image. No planner or action computation.

Two real RGB cameras stay untouched. This THIRD image re-expresses the same
raw64 pre-action geometry/contact measurements in a readable fixed world frame.
For touch, green denotes the measured object top. Otherwise it is the desired
TARGET AABB CENTER in its current orientation: current AABB center +
(object-base goal - current object base). No EE waypoint, teacher phase,
future label, trajectory or motor command is produced.
"""
import numpy as np

from VLA.scene64_conditioning import validate_scene

SCHEMA = 'simGT_third_visual_worldXY_height_contact_v3_r53'
SIZE = 384
XY_BOUNDS = (-.42, .42, -.80, -.20)
RECT = (16, 40, 368, 280)
TARGET, GOAL, PAD = (255, 78, 82), (76, 235, 124), (68, 210, 255)
FONT = {
 'A':(14,17,17,31,17,17,17),'C':(14,17,16,16,16,17,14),
 'D':(30,17,17,17,17,17,30),'E':(31,16,16,30,16,16,31),
 'G':(14,17,16,23,17,17,15),'H':(17,17,17,31,17,17,17),
 'L':(16,16,16,16,16,16,31),'N':(17,25,21,19,17,17,17),
 'O':(14,17,17,17,17,17,14),'P':(30,17,17,30,16,16,16),
 'R':(30,17,17,30,20,18,17),'T':(31,4,4,4,4,4,4),
 'W':(17,17,17,21,21,21,10),'Z':(31,1,2,4,8,16,31),
 '0':(14,17,19,21,25,17,14),'1':(4,12,4,4,4,4,14),
 '2':(14,17,1,2,4,8,31),'3':(30,1,1,14,1,1,30),
 '4':(2,6,10,18,31,2,2),'5':(31,16,16,30,1,1,30),
 '6':(14,16,16,30,17,17,14),'7':(31,1,2,4,8,8,8),
 '8':(14,17,17,14,17,17,14),'9':(14,17,17,15,1,1,14),
 '.':(0,0,0,0,0,6,6),'-':(0,0,0,31,0,0,0),' ':(0,)*7}


def contract():
    return {'schema': SCHEMA, 'image_shape': [384,384,3], 'input_order': ['front','wrist','privileged_map'],
        'front_wrist_original_RGB_unchanged': True, 'world_X_right_Y_up': True,
        'world_XY_bounds_m': list(XY_BOUNDS), 'height_range_m': [0., .72],
        'target_rgb': list(TARGET), 'goal_rgb': list(GOAL), 'pad_rgb': list(PAD),
        'goal_semantics': 'touch_measured_object_top_OTHER_tasks_target_center_equivalent_NOT_gripper_waypoint',
        'large_coordinate_markers_and_binary_contact_tiles_for_native224': True,
        'values_only_from_pre_action_raw64': True, 'no_teacher_phase_action_reward_or_success_input': True,
        'all_motor_commands_still_genuine_native_neural_outputs': True,
        'method_label': 'VLA_plus_simulator_ground_truth_visual_and_numeric_observations_NOT_vanilla_VLA'}


def task_from_instruction(instruction):
    words=str(instruction).strip().lower().split()
    mapping={'touch':'touch','pick':'pick','move':'move','put':'pick_place'}
    if not words or words[0] not in mapping:
        raise ValueError('Original task instruction required; no geometric phase fallback')
    return mapping[words[0]]


def geometry(raw64, task):
    if task not in ('touch','pick','move','pick_place'):
        raise ValueError('One of the four original language tasks is required')
    raw = validate_scene(raw64)
    pad = raw[7:10].astype(np.float64)
    target = pad + raw[11:14]
    goal = target + (np.array([0.,0.,raw[26]/2]) if task == 'touch' else raw[14:17])
    neighbors = raw[40:].reshape(6,4)
    return raw, pad, target, goal, neighbors


def project(xy):
    x0,x1,y0,y1 = XY_BOUNDS
    l,t,r,b = RECT
    xy = np.asarray(xy, np.float64)
    pixel = np.array([l+(xy[0]-x0)*(r-l)/(x1-x0), b-(xy[1]-y0)*(b-t)/(y1-y0)])
    return np.rint(np.clip(pixel, [l,t], [r,b])).astype(int)


def text(canvas, value, x, y, color, scale=2):
    for char in value:
        for j, row in enumerate(FONT[char]):
            for i in range(5):
                if row & (1 << (4-i)):
                    canvas[y+j*scale:y+(j+1)*scale, x+i*scale:x+(i+1)*scale] = color
        x += 6*scale


def render(raw64, task):
    raw, pad, target, goal, neighbors = geometry(raw64,task)
    image = np.full((SIZE,SIZE,3), (18,21,27), np.uint8)
    l,t,r,b = RECT
    image[t:b+1,l:r+1] = (35,39,46)
    yy,xx = np.ogrid[:SIZE,:SIZE]
    def dot(point, radius, color):
        x,y = point
        image[(xx-x)**2+(yy-y)**2 <= radius**2] = color
    def cross(point, radius, color, width=2):
        x,y = point
        image[max(0,y-radius):min(SIZE,y+radius+1), max(0,x-width//2):min(SIZE,x+width//2+1)] = color
        image[max(0,y-width//2):min(SIZE,y+width//2+1), max(0,x-radius):min(SIZE,x+radius+1)] = color
    for x in np.arange(-.4,.401,.1):
        u,_ = project((x,-.5));image[t:b+1,u] = (55,60,69)
    for y in np.arange(-.8,-.199,.1):
        _,v = project((0,y));image[v,l:r+1] = (55,60,69)
    sx,sy = (r-l)/(.84), (b-t)/(.60)
    for dx,dy,dz,radius in neighbors:
        if radius == 0:
            continue
        p = project(pad[:2]+[dx,dy])
        ellipse = ((xx-p[0])/max(1.,radius*sx))**2 + ((yy-p[1])/max(1.,radius*sy))**2 <= 1
        ellipse &= (xx >= l) & (xx <= r) & (yy >= t) & (yy <= b)
        image[ellipse] = (114,121,136)
    p = project(target[:2])
    rx,ry = max(3,int(round(raw[24]*sx/2))),max(3,int(round(raw[25]*sy/2)))
    rectangle = (abs(xx-p[0]) <= rx) & (abs(yy-p[1]) <= ry)
    inside = (abs(xx-p[0]) < max(1,rx-2)) & (abs(yy-p[1]) < max(1,ry-2))
    image[rectangle & ~inside & (xx >= l) & (xx <= r) & (yy >= t) & (yy <= b)] = TARGET
    dot(p,8,TARGET)
    cross(project(goal[:2]),12,GOAL,3)
    cross(project(pad[:2]),10,PAD,3)
    # An off-map marker must not silently look like an in-bounds position.
    x0,x1,y0,y1 = XY_BOUNDS
    for point,color in ((pad,PAD),(target,TARGET),(goal,GOAL)):
        if point[0] < x0 or point[0] > x1 or point[1] < y0 or point[1] > y1:
            u,v = project(point[:2]);cross((u,v),10,color,4)
    for value,x,color in [('TARGET',8,TARGET),('GOAL',146,GOAL),('PAD',282,PAD)]:
        text(image,value,x,9,color)
    for i,(label,flag) in enumerate(zip(('L','R','H','W','A'),raw[[28,29,17,30,31]],strict=True)):
        x = 8+i*75
        color=(245,99,88) if label == 'W' and flag else (88,230,120) if flag else (60,65,74)
        image[288:309,x:x+68]=color
        text(image,label,x+28,291,(15,20,25) if flag else (205,210,220),2)
    for i,(label,value,color) in enumerate((('T',target[2],TARGET),('G',goal[2],GOAL),('P',pad[2],PAD))):
        y = 316+i*22
        text(image,label+' Z',12,y,color,1)
        image[y:y+9,58:284] = (48,53,62)
        endpoint = 58+int(round(np.clip(value/.72,0,1)*226))
        image[y:y+9,58:endpoint] = color
        text(image,f'{value:.2f}',300,y,color,1)
    return image
