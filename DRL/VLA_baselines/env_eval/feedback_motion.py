"""Task-conditioned C2 action chunks and a jerk-bounded joint reference.

These are model-based control components, not a trained neural policy. Design
references and the distinction are recorded in the accompanying survey.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.interpolate import BPoly
from scipy.spatial.transform import Rotation
from ruckig import Ruckig, InputParameter, OutputParameter, Result, Synchronization
from env_eval.oracle_trajectory import minimum_jerk
from env_eval.oracle_style import blend_quaternion,styled_approach,styled_carry


@dataclass(frozen=True)
class MotionProfile:
    tempo: float = 1.0
    approach_scale: float = 1.0
    joint_jerk: float = 35.0
    contact_close_seconds: float = .58
    contact_distance: float = .009
    corridor_margin: float = .025
    wrist_compensation: float = .025

    def to_dict(self):
        return asdict(self)


class CorrectionEnsemble:
    """Overlap model-based predictions for the same absolute future instant.

    The predictor is observed displacement/velocity, not an ACT network. Newer
    predictions receive more weight to retain feedback responsiveness.
    """
    def __init__(self,horizon=.25,time_constant=.09):
        self.horizon=horizon;self.time_constant=time_constant;self.chunks=[]

    def add(self,time,offset,velocity):
        self.chunks.append((float(time),np.array(offset),np.array(velocity)))

    def sample(self,time):
        self.chunks=[c for c in self.chunks if 0<=time-c[0]<=self.horizon+1e-9]
        if not self.chunks:return np.zeros(3)
        age=np.array([time-c[0] for c in self.chunks])
        weights=np.exp(-age/self.time_constant)
        predictions=np.array([offset+velocity*dt for (_,offset,velocity),dt in zip(self.chunks,age)])
        return np.average(predictions,axis=0,weights=weights)


class JointReference:
    """Continuously replan from the current commanded q, qdot and qddot.

    A small causal ensemble rejects IK/contact noise before online generation.
    Feed-forward from its velocity compensates its phase lag. No joint, body or
    object pose is reset while executing a motion.
    """
    def __init__(self, position, dt, max_jerk=35.):
        self.dt=dt
        self.generator=Ruckig(len(position),dt)
        self.input=InputParameter(len(position));self.output=OutputParameter(len(position))
        self.input.current_position=list(position)
        self.input.current_velocity=[0.]*len(position)
        self.input.current_acceleration=[0.]*len(position)
        self.input.max_velocity=[1.15]*len(position)
        self.input.max_acceleration=[2.5]*len(position)
        self.input.max_jerk=[float(max_jerk)]*len(position)
        # Match the complete future state at one common preview time. The
        # nominal pose clock must first respect the wrist's velocity limits.
        self.input.synchronization=Synchronization.Time
        self.filtered=np.array(position,dtype=float)
        self.velocity=np.zeros(len(position))

    def step(self, target, future=None):
        previous=self.filtered.copy()
        self.filtered += (1-np.exp(-self.dt/.020))*(np.asarray(target)-self.filtered)
        derivative=(self.filtered-previous)/self.dt
        self.velocity += (1-np.exp(-self.dt/.045))*(derivative-self.velocity)
        velocity=np.clip(self.velocity,-.85,.85)
        if future is None:
            velocity[np.abs(velocity)<1e-8]=0.
            self.input.target_position=(self.filtered+.050*velocity).tolist()
            self.input.target_velocity=velocity.tolist()
            self.input.target_acceleration=[0.]*len(velocity)
            self.input.minimum_duration=0.
        else:
            position,velocity,acceleration,horizon=future
            self.input.target_position=position.tolist()
            self.input.target_velocity=np.clip(velocity,-1.05,1.05).tolist()
            self.input.target_acceleration=np.clip(acceleration,-2.,2.).tolist()
            self.input.minimum_duration=horizon
        result=self.generator.update(self.input,self.output)
        if result not in (Result.Working,Result.Finished):
            raise RuntimeError(f'Joint reference generation failed: {result}')
        self.output.pass_to_input(self.input)
        return (np.array(self.output.new_position),np.array(self.output.new_velocity),
                np.array(self.output.new_acceleration))


class ActionChunk:
    """Quintic pieces share position, velocity and acceleration at each knot."""
    def __init__(self, points, times, velocities, q0, q1, *, rotation_end=.72,
                 transport_axis=None, wrist_compensation=0.):
        points=np.asarray(points,dtype=float);times=np.asarray(times,dtype=float)
        self.duration=float(times[-1]);self.points=points
        derivatives=[[point,np.asarray(velocity),np.zeros(3)] for point,velocity in zip(points,velocities)]
        self.position=BPoly.from_derivatives(times,derivatives)
        self.q0=np.asarray(q0);self.q1=np.asarray(q1);self.rotation_end=rotation_end
        self.transport_axis=transport_axis;self.wrist_compensation=wrist_compensation

    def sample(self, elapsed):
        elapsed=float(np.clip(elapsed,0,self.duration));s=elapsed/self.duration
        turn=float(minimum_jerk(np.clip(s/self.rotation_end,0,1)))
        q=blend_quaternion(self.q0,self.q1,turn)
        if self.transport_axis is not None:
            # Tilt is aligned with travel acceleration, vanishes at both ends,
            # and changes sign during braking; no random decorative wrist wiggle.
            direction=np.asarray(self.transport_axis)
            magnitude=self.wrist_compensation*np.sin(2*np.pi*s)*np.sin(np.pi*s)**2
            q=(Rotation.from_rotvec(np.cross([0,0,1],direction)*magnitude)*Rotation.from_quat(q)).as_quat()
        return self.position(elapsed),q


def approach_chunk(start,goal,q0,q1,duration):
    # Preserve the feasible approach corridor and arc-length clock. Orientation
    # and translation must stay coordinated: forcing the wrist turn too early
    # can saturate joint limits even when the Cartesian curve looks smooth.
    curve=styled_approach(start,goal,q0,q1,duration=duration,side=0.,lean=0.)
    origin=Rotation.from_quat(q0);turn=(origin.inv()*Rotation.from_quat(q1)).as_rotvec()
    def sample(elapsed):
        normalized=float(np.clip(elapsed/duration,0,1))
        clock=float(minimum_jerk(normalized))
        parameter=float(curve.inverse_length(clock)) if curve.inverse_length is not None else clock
        rotation=float(minimum_jerk(np.clip(normalized/.88,0,1)))
        return curve.geometry(np.clip(parameter,0,1)),(origin*Rotation.from_rotvec(rotation*turn)).as_quat()
    curve.sample=sample
    return curve


def transport_chunk(start,goal,q0,q1,clearance,duration,*,lift=False,wrist=.025):
    start,goal=np.asarray(start),np.asarray(goal);delta=goal-start
    horizontal=np.r_[delta[:2],0.];length=np.linalg.norm(horizontal)
    direction=horizontal/max(length,1e-9)
    if lift:
        # Position, speed and curvature derive from the displacement, with no
        # seeded lateral ornament. A small approach-direction carryover is
        # handled by the joint reference rather than a second arbitrary bend.
        return ActionChunk([start,goal],[0,duration],[np.zeros(3),np.zeros(3)],q0,q1,
                           rotation_end=1.)
    curve=styled_carry(start,goal,q0,clearance=clearance,duration=duration,side=0.,lean=0.,goal_quaternion=q1)
    position_sample=curve.sample
    def sample(elapsed):
        point,_=position_sample(elapsed);s=float(np.clip(elapsed/duration,0,1))
        quaternion=blend_quaternion(q0,q1,float(minimum_jerk(s)))
        magnitude=wrist*np.sin(2*np.pi*s)*np.sin(np.pi*s)**2
        quaternion=(Rotation.from_rotvec(np.cross([0,0,1],direction)*magnitude)*Rotation.from_quat(quaternion)).as_quat()
        return point,quaternion
    curve.sample=sample
    return curve


def retract_chunk(start,goal,quaternion,duration):
    start,goal=np.asarray(start),np.asarray(goal)
    return ActionChunk([start,goal],[0,duration],[np.zeros(3),np.zeros(3)],quaternion,quaternion)
