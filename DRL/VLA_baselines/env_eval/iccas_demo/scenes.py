"""Deterministic shared layouts; all names are unique physical scene entities."""
from dataclasses import dataclass
from VLA.common.tasks import TaskSpec,SceneObjectSpec


@dataclass(frozen=True)
class Step:
    action:str
    pick:str
    place:str|None=None


@dataclass(frozen=True)
class RequestedScene:
    sid:int
    objects:tuple
    steps:tuple

    def task(self,index=0):
        step=self.steps[index];positions=dict(self.objects)
        action={'put':'pick_place'}.get(step.action,step.action)
        prompt=f'{step.action.capitalize()} the {step.pick}'
        if step.place:prompt+=(' onto ' if step.action=='move' else ' into ')+step.place
        elif step.action=='pick':prompt+=' up'
        others=tuple(SceneObjectSpec(name,(*xy,.06),name.endswith((' bowl',' cup')),'requested_ICCAS_object')
            for name,xy in self.objects if name not in (step.pick,step.place))
        return TaskSpec(('touch','pick','move','put').index(step.action),action,prompt,
            step.pick,(*positions[step.pick],.06),step.place,
            (*positions[step.place],.06) if step.place else None,
            max_steps=10000,scene_profile='ICCAS_demo_requested10',scene_objects=others)


def scene(sid,pairs,steps):
    return RequestedScene(sid,tuple(pairs),tuple(Step(*x) for x in steps))


SCENES=(
 scene(1,[('yellow tennis_ball',(-.13,-.53)),('yellow bowl',(.15,-.43)),
   ('red block',(-.27,-.31)),('orange orange',(-.25,-.72)),('yellow block',(.27,-.69))],
   [('put','yellow tennis_ball','yellow bowl')]),
 scene(2,[('orange orange',(-.13,-.53)),('blue bowl',(.15,-.43)),
   ('yellow bowl',(-.25,-.31)),('blue block',(-.27,-.73)),
   ('blue cup',(.27,-.69)),('brown bottle',(.03,-.25))],
   [('put','orange orange','blue bowl')]),
 scene(3,[('red block',(-.13,-.53)),('blue block',(.15,-.48)),
   ('yellow block',(-.26,-.29)),('green bowl',(.23,-.28)),
   ('blue cup',(-.27,-.73)),('red bowl',(.23,-.72))],
   [('move','red block','blue block')]),
 scene(4,[('red apple',(-.13,-.53)),('blue cup',(.15,-.43)),
   ('red bowl',(-.25,-.30)),('red block',(-.27,-.72)),
   ('orange orange',(.27,-.70)),('green chalk_brush',(.04,-.28))],
   [('put','red apple','blue cup')]),
 scene(5,[('red block',(-.13,-.53)),('orange bowl',(.17,-.43)),
   ('blue block',(-.26,-.30)),('blue block 2',(.02,-.29)),
   ('green bowl',(.24,-.70)),('orange block',(-.26,-.72))],
   [('pick','red block')]),
 scene(6,[('blue block',(-.13,-.53)),('green bowl',(.17,-.43)),
   ('red block',(-.26,-.30)),('red bowl',(.24,-.72)),
   ('orange bowl',(-.25,-.72)),('red pen',(.025,-.29))],
   [('touch','blue block')]),
 scene(7,[('yellow tennis_ball',(-.13,-.53)),('red bowl',(.15,-.43)),
   ('green bowl',(-.26,-.30)),('orange orange',(-.25,-.72)),
   ('red block',(.27,-.71)),('orange block',(.26,-.28)),('yellow block',(.00,-.28))],
   [('put','yellow tennis_ball','red bowl')]),
 scene(8,[('orange orange',(-.17,-.54)),('yellow tennis_ball',(.0,-.56)),
   ('red apple',(.17,-.55)),('orange bowl',(-.22,-.30)),
   ('green bowl',(.0,-.30)),('red bowl',(.22,-.30)),('blue bowl',(.28,-.74))],
   [('put','orange orange','orange bowl'),('put','yellow tennis_ball','green bowl'),
    ('put','red apple','red bowl')]),
 scene(9,[('red block',(-.24,-.56)),('blue block',(-.08,-.56)),
   ('yellow block',(.08,-.56)),('orange block',(.24,-.56)),
   ('red bowl',(-.27,-.30)),('blue bowl',(-.09,-.30)),
   ('yellow bowl',(.09,-.30)),('orange bowl',(.27,-.30)),
   ('green chalk_brush',(-.24,-.75)),('green bowl',(.09,-.75))],
   [('put','red block','red bowl'),('put','blue block','blue bowl'),
    ('put','yellow block','yellow bowl'),('put','orange block','orange bowl')]),
 scene(10,[('green chalk_brush',(-.13,-.54)),('green bowl',(.14,-.48)),
   ('orange block',(-.29,-.72)),('red block',(-.10,-.74)),
   ('blue block',(.08,-.74)),('yellow block',(.27,-.73)),
   ('orange bowl',(-.27,-.29)),('red bowl',(-.09,-.29)),
   ('blue bowl',(.09,-.29)),('yellow bowl',(.27,-.29))],
   [('put','green chalk_brush','green bowl')]),
)


def get_scene(sid):return SCENES[sid-1]


def display_command(step):
    aliases={'yellow tennis_ball':'tennis ball','orange orange':'orange','red apple':'apple',
        'green chalk_brush':'chalk brush'}
    pick=aliases.get(step.pick,step.pick);place=aliases.get(step.place,step.place)
    text=f'{step.action.capitalize()} {pick}'
    if place:text+=(' onto ' if step.action=='move' else ' into ')+place
    elif step.action=='pick':text+=' up'
    return text
