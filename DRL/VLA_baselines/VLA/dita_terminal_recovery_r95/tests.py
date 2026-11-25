import copy
import os
import signal
import subprocess
import sys
import unittest
from types import SimpleNamespace

from .identity import argv_equal, await_identity


class Registration(unittest.TestCase):
    def setUp(self):
        self.cmd=['python','-u','-m','approved.worker']
        self.record={'pid':123,'ppid':456,'start_ticks':'789','command':self.cmd}
        self.child=SimpleNamespace(pid=123,poll=lambda:None)

    def test_waits_for_exact_post_exec_argv(self):
        transient={**self.record,'command':['python','-m','parent.coordinator']}
        values=iter([transient,self.record,copy.deepcopy(self.record)])
        actual,history=await_identity(self.child,self.cmd,456,lambda _:next(values),pause=lambda _:None)
        self.assertEqual(actual,self.record); self.assertEqual(len(history),2)

    def test_wrong_parent_fails_closed(self):
        with self.assertRaises(ValueError):
            await_identity(self.child,self.cmd,456,lambda _:{**self.record,'ppid':999},pause=lambda _:None)

    def test_pid_start_ticks_change_fails_closed(self):
        values=iter([self.record,{**self.record,'start_ticks':'790'}])
        with self.assertRaises(ValueError):
            await_identity(self.child,self.cmd,456,lambda _:next(values),pause=lambda _:None)

    def test_early_exit_is_not_registered(self):
        child=SimpleNamespace(pid=123,poll=lambda:1)
        with self.assertRaises(RuntimeError): await_identity(child,self.cmd,456,lambda _:self.record)

    def test_no_partial_argv_match(self):
        self.assertFalse(argv_equal(self.cmd,self.cmd+['fit']))

    def test_unknown_argv_times_out(self):
        times=iter([0.,.1,.2,.3,.4])
        with self.assertRaises(TimeoutError):
            await_identity(self.child,self.cmd,456,lambda _:{**self.record,'command':['unrelated']},
                timeout=.25,clock=lambda:next(times),pause=lambda _:None)

    def test_actual_benign_child_full_identity(self):
        from VLA.continuous_r86.owner import identity
        command=[sys.executable,'-c','import time; time.sleep(0.15)']
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        try:
            actual,_=await_identity(child,command,os.getpid(),identity)
            self.assertEqual(actual['pid'],child.pid)
        finally: child.wait(timeout=5)

    def test_native_fit_still_forbidden_all_roles(self):
        from VLA.dita_close800_r94.common import authorize_signal,CP
        record={'command':['python','-m','VLA.recovery_r85.dita_train','fit','--output',str(CP)]}
        for role in ('own_reload','own_benchmark','queue_parent','idle_recovery'):
            with self.assertRaises(ValueError): authorize_signal(record,signal.SIGTERM,role)


if __name__=='__main__': unittest.main()
