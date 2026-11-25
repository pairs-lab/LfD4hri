"""Skip files, NOT the renderer/mask calls that may affect later observations."""


class DiscardWriter:
    def __init__(self,*args,**kwargs):self.frames=0;self.closed=False
    def __bool__(self):return True
    def append_data(self,image):
        if self.closed:raise ValueError('Closed discard writer')
        self.frames+=1
    def close(self):self.closed=True


def discard_snapshot(output_dir,task_name,observation,env):
    # Retain the actual native snapshot's segmentation calls in their original
    # order, but omit PIL/JSON file writes and pure local image statistics.
    if hasattr(env,'target_mask'):
        for camera in ('front','wrist'):
            for role in ('pick','place'):env.target_mask(camera,role)
