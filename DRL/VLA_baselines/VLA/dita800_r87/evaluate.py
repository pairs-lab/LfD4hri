"""Reuse the unchanged R83 physical loop; explicit user override of SR gate.

Do NOT falsely set frozen_after_confirmation=True. The only in-memory AST
substitution replaces its eligibility boolean with an explicit authorized
frozen-benchmark check. Observation/action/physics/success code stays exact.
"""
import argparse
import ast
import copy
import inspect
from pathlib import Path
from VLA.common.artifact_io import atomic_json
from VLA.recovery_r85.evaluate import atomic_json as safe_atomic_json


def authorized_frozen(method,cohort):
    authorization=method.get('benchmark_authorization',{})
    return (method.get('frozen_for_user_requested_benchmark') is True
        and authorization.get('user_requested_800_despite_failed_pilot_gate') is True
        and method.get('frozen_after_confirmation') is False
        and cohort['role']=='fresh_secondary_final800'
        and cohort['seeds']==list(range(20284001,20284201)))


def compile_unchanged_physical_loop(old):
    source=inspect.getsource(old.main);tree=ast.parse(source)
    class EligibilityOnly(ast.NodeTransformer):
        count=0
        def visit_Call(self,node):
            self.generic_visit(node)
            if (isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name)
                and node.func.value.id=='method' and node.func.attr=='get' and len(node.args)==1
                and isinstance(node.args[0],ast.Constant) and node.args[0].value=='frozen_after_confirmation'):
                self.count+=1
                return ast.copy_location(ast.Call(func=ast.Name(id='authorized_frozen',ctx=ast.Load()),
                    args=[ast.Name(id='method',ctx=ast.Load()),ast.Name(id='cohort',ctx=ast.Load())],keywords=[]),node)
            return node
    edit=EligibilityOnly();tree=ast.fix_missing_locations(edit.visit(copy.deepcopy(tree)))
    if edit.count!=1:raise ValueError('Frozen evaluator guard changed: refuse a broad source rewrite')
    namespace={**old.__dict__,'authorized_frozen':authorized_frozen,'atomic_json':safe_atomic_json}
    exec(compile(tree,'<R87 one explicit eligibility override; R83 physics unchanged>','exec'),namespace)
    return namespace['main']


def main(a):
    from VLA.autonomy_r83 import evaluate as old
    from VLA.autonomy_r83.common import read
    if not authorized_frozen(read(a.method),read(a.cohort)):
        raise ValueError('This runner only accepts the explicitly requested frozen Dita1024 cohort800')
    compile_unchanged_physical_loop(old)(a)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('method','cohort','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--port',type=int,required=True);main(p.parse_args())
