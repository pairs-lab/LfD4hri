"""Disclosed controller faults, not learned-RL weaknesses; same paired plant."""
import argparse
from pathlib import Path
import numpy as np
from VLA.qualitative_r57 import oracle as base


class ControlledOracle(base.ScriptedOracle):
    def step(self,center,q,command):
        if self.fault=='persistent_miss' and self.phase in ('approach','grasp'):
            scale=1. if self.phase=='grasp' else float(np.clip((self.phase_progress-.35)/.30,0,1))
            center=np.asarray(center).copy();center[0]+=.075*scale
            if scale and not self.fault_events:self.fault_events.append({
                'type':'scripted_persistent_target_offset_during_approach_AND_grasp',
                'time_s':self.tick_count/self.env.physics_hz,'offset_x_m':.075,
                'not_a_ground_slam_without_measured_contact':True})
        if self.fault=='closed_jaw_approach' and self.phase in ('approach','grasp'):
            command=1.
            if not self.fault_events:self.fault_events.append({
                'type':'scripted_closed_jaws_during_object_approach','time_s':self.tick_count/self.env.physics_hz,
                'actual_contact_displacement_and_failure_must_be_audited':True})
        return super().step(center,q,command)

    def curve(self,curve,phase,*args,**kwargs):
        if self.fault=='closed_jaw_approach' and phase=='approach':args=(1.,1.);kwargs={}
        return super().curve(curve,phase,*args,**kwargs)

    def run(self):
        try:return super().run()
        except RuntimeError as error:
            if ('No verified bilateral grasp' in str(error) and not self.env.safety_latched
                    and self.fault in ('persistent_miss','closed_jaw_approach')):
                # Real physics at the actual failed pose, not a frozen frame,
                # exposes empty closure/aftermath. Never continue past safety.
                self.hold(self.env.gripper.grasp_center(),self.env.pose()[1],1.5,'fault_aftermath',1.)
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--paired-capture',type=Path,required=True)
    parser.add_argument('--fault',choices=('none','drop_in_transit','slow_transport_timeout',
        'closed_jaw_approach','persistent_miss'),required=True)
    base.ScriptedOracle=ControlledOracle
    base.main(parser.parse_args())
