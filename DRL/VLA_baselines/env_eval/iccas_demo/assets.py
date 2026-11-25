"""Native assets plus explicit physical procedural props, never image overlays."""
from pathlib import Path
import hashlib
import os
import numpy as np
import pybullet as p
from config import COLORS
from env_eval.feedback_environment import FeedbackEnvironment


def mesh_visual(vertices,faces,color):
    # Bullet's EGL plugin cannot load inline GEOM_MESH visual vertices on
    # this build. A normal OBJ asset is supported by both EGL and TinyRenderer.
    content=''.join('v '+' '.join(f'{float(v):.12g}' for v in row)+'\n' for row in vertices)
    content+=''.join('f '+' '.join(str(int(v)+1) for v in faces[i:i+3])+'\n' for i in range(0,len(faces),3))
    digest=hashlib.sha256(content.encode()).hexdigest()
    directory=Path(__file__).resolve().parents[2]/'eval/ICCAS_demo/_generated_meshes';directory.mkdir(parents=True,exist_ok=True)
    path=directory/(digest+'.obj')
    if not path.exists():
        temporary=path.with_suffix(f'.{os.getpid()}.tmp');temporary.write_text(content);temporary.replace(path)
    return p.createVisualShape(p.GEOM_MESH,fileName=str(path),meshScale=(1,1,1),rgbaColor=color)


def compound(parts,color,position,mass=0.):
    types=[x[0] for x in parts];xyz=[x[1] for x in parts]
    quats=[x[3] if len(x)>3 else (0,0,0,1) for x in parts]
    half=[x[2] if x[0]==p.GEOM_BOX else (1,1,1) for x in parts]
    radii=[x[2][0] if x[0]==p.GEOM_CYLINDER else 1 for x in parts]
    lengths=[x[2][1] if x[0]==p.GEOM_CYLINDER else 1 for x in parts]
    collision=p.createCollisionShapeArray(shapeTypes=types,halfExtents=half,radii=radii,lengths=lengths,
        collisionFramePositions=xyz,collisionFrameOrientations=quats)
    visual=p.createVisualShapeArray(shapeTypes=types,halfExtents=half,radii=radii,lengths=lengths,
        visualFramePositions=xyz,visualFrameOrientations=quats,rgbaColors=[color]*len(parts))
    return p.createMultiBody(mass,collision,visual,basePosition=position)


def cup(position):
    # A real open cavity, not a solid collision cylinder. 100 mm opening fits
    # the native apple. The handle stays outside the receiving cavity.
    inner=.050;wall=.004;height=.085;n=12
    parts=[(p.GEOM_CYLINDER,(0,0,.003),(inner+wall,.006))]
    for angle in np.linspace(0,2*np.pi,n,endpoint=False):
        radius=inner+wall/2
        parts.append((p.GEOM_BOX,(radius*np.cos(angle),radius*np.sin(angle),height/2),
            (wall/2, radius*np.tan(np.pi/n)+.0004,height/2),p.getQuaternionFromEuler((0,0,float(angle)))))
    # Keep Bullet compound shape arrays below their native 16-child limit.
    # A separate smooth visual mesh includes the actual hollow wall and handle.
    types=[x[0] for x in parts]
    collision=p.createCollisionShapeArray(shapeTypes=types,
        halfExtents=[x[2] if x[0]==p.GEOM_BOX else (1,1,1) for x in parts],
        radii=[inner+wall]+[1.]*n,lengths=[.006]+[1.]*n,
        collisionFramePositions=[x[1] for x in parts],
        collisionFrameOrientations=[x[3] if len(x)>3 else (0,0,0,1) for x in parts])
    vertices=[];faces=[];count=64
    for radius,z in ((inner+wall,0),(inner+wall,height),(inner,.006),(inner,height)):
        vertices.extend((radius*np.cos(a),radius*np.sin(a),z) for a in np.linspace(0,2*np.pi,count,endpoint=False))
    def quad(a,b,c,d):faces.extend((a,b,c,a,c,d))
    for i in range(count):
        j=(i+1)%count
        quad(i,j,count+j,count+i)
        quad(2*count+i,3*count+i,3*count+j,2*count+j)
        quad(count+i,count+j,3*count+j,3*count+i)
    bottom=len(vertices);vertices.append((0,0,.006))
    for i in range(count):faces.extend((bottom,2*count+i,2*count+(i+1)%count))
    # Tubular rounded handle, in the XZ plane, built into the visual mesh.
    origin=len(vertices);segments=32;tube=10
    for angle in np.linspace(-np.pi/2,np.pi/2,segments):
        for t in np.linspace(0,2*np.pi,tube,endpoint=False):
            vertices.append((inner+.003+(.024+.004*np.cos(t))*np.cos(angle),
                .004*np.sin(t),.045+(.028+.004*np.cos(t))*np.sin(angle)))
    for i in range(segments-1):
        for j in range(tube):quad(origin+i*tube+j,origin+i*tube+(j+1)%tube,
            origin+(i+1)*tube+(j+1)%tube,origin+(i+1)*tube+j)
    visual=mesh_visual(vertices,faces,COLORS['blue'])
    return p.createMultiBody(0,collision,visual,basePosition=position)


def bottle(position):
    parts=[(p.GEOM_CYLINDER,(0,0,.046),(.022,.092)),
           (p.GEOM_CYLINDER,(0,0,.101),(.012,.026)),
           (p.GEOM_CYLINDER,(0,0,.116),(.013,.007))]
    return compound(parts,COLORS['brown'],position,.09)


def brush(position):
    # The user's permitted green trapezoid, with identical convex visual and
    # collision geometry. Thin enough to fit a Robotiq body-level pinch.
    xy=[(-.031,-.021),(.031,-.021),(.024,.021),(-.024,.021)]
    vertices=[(x,y,z) for z in (-.013,.013) for x,y in xy]
    faces=[0,2,1,0,3,2,4,5,6,4,6,7]
    for i in range(4):
        j=(i+1)%4;faces += [i,j,j+4,i,j+4,i+4]
    collision=p.createCollisionShape(p.GEOM_MESH,vertices=vertices,indices=faces)
    visual=mesh_visual(vertices,faces,COLORS['green'])
    body=p.createMultiBody(.045,collision,visual,basePosition=position)
    # Inline dynamic mesh construction defaults to zero inertia in this build.
    # Use a conservative physical box inertia for the 62×42×26 mm body.
    inertia=.045/12*np.array([.042**2+.026**2,.062**2+.026**2,.062**2+.042**2])
    p.changeDynamics(body,-1,localInertiaDiagonal=inertia.tolist())
    return body


def pen(position):
    rot=p.getQuaternionFromEuler((0,np.pi/2,0))
    return compound([(p.GEOM_CYLINDER,(0,0,0),(.005,.104),rot),
        (p.GEOM_CYLINDER,(.053,0,0),(.0027,.012),rot)],COLORS['red'],position,.015)


class ICCASEnvironment(FeedbackEnvironment):
    def __init__(self,*args,**kwargs):
        self.asset_manifest={};super().__init__(*args,**kwargs)

    def _spawn_object(self,name,position,fixed):
        if name=='blue cup':body=cup(position);source='procedural hollow cup, 100mm inner diameter'
        elif name=='brown bottle':body=bottle(position);source='procedural brown cylinder body/neck/cap'
        elif name=='green chalk_brush':body=brush(position);source='user-authorized green trapezoid'
        elif name=='red pen':body=pen(position);source='procedural slender red cylinder and nib'
        else:
            native='blue block' if name=='blue block 2' else name
            body=super()._spawn_object(native,position,fixed);source='unchanged native asset: '+native
        # Initial layout coordinates denote physical AABB centres, independent
        # of each native URDF's inertial offset. Only initial loading is moved.
        lo,hi=map(np.asarray,p.getAABB(body));base,quat=p.getBasePositionAndOrientation(body)
        base=np.array(base);base[:2]+=np.asarray(position[:2])-(lo[:2]+hi[:2])/2
        base[2]+=p.getAABB(self.table_id)[1][2]+.001-lo[2]
        p.resetBasePositionAndOrientation(body,base,quat)
        p.changeDynamics(body,-1,lateralFriction=1.2,spinningFriction=.025,
            rollingFriction=.02,restitution=0.,linearDamping=.3,angularDamping=.3)
        self.asset_manifest[name]={'source':source,'body_id':body,'spawn_AABB':[list(x) for x in p.getAABB(body)]}
        return body
