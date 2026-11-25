"""CPU-only actual cross-Python openpi websocket roundtrip, NOT VLA inference."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time

import numpy as np
from PIL import Image

from VLA.cross_vla120_trial import atomic_json
from VLA.run_cross_vla120 import OPENPI_PYTHON, ROOT, TASKS, SEEDS

SERVER = '''
import sys
from openpi_client import msgpack_numpy
from websockets.sync.server import serve
def handler(connection):
    connection.send(msgpack_numpy.packb({"transport_echo_only": True, "model_inference_used": False}))
    for message in connection:
        payload = msgpack_numpy.unpackb(message)
        connection.send(msgpack_numpy.packb(payload))
with serve(handler, "127.0.0.1", int(sys.argv[1]), compression=None, max_size=None) as server:
    server.serve_forever()
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8040)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Transport audit must not overwrite earlier evidence")
    args.output.mkdir(parents=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", args.port))
    from openpi_client import image_tools, websocket_client_policy
    server = None
    with (args.output / "server.log").open("x") as log:
        try:
            server = subprocess.Popen([str(OPENPI_PYTHON), "-c", SERVER, str(args.port)], cwd=ROOT,
                                      stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 30
            while True:
                if server.poll() is not None:
                    raise RuntimeError("Owned transport-only server failed")
                try:
                    with socket.create_connection(("127.0.0.1", args.port), timeout=1):
                        break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Owned transport-only startup expired")
                    time.sleep(.2)
            client = websocket_client_policy.WebsocketClientPolicy(host="127.0.0.1", port=args.port)
            if client.get_server_metadata().get("model_inference_used") is not False:
                raise ValueError("This audit must NEVER invoke/score a model")
            rows = []
            for task in TASKS:
                directory = args.scene_root / f"seed{SEEDS[0]}" / task
                reset = json.loads((directory / "reset.json").read_text())
                payload = {"prompt": reset["task"]["prompt"],
                    "joints": ((np.asarray(reset["joint_positions_rad"], dtype=np.float32) + np.pi) % (2 * np.pi) - np.pi).astype(np.float32),
                    "gripper": np.asarray([reset["gripper_command_closure"]], dtype=np.float32),
                    "__policy_noise__": np.random.default_rng(20261013).standard_normal((50, 32), dtype=np.float32)}
                for camera, key in (("front", "base_rgb"), ("wrist", "wrist_rgb")):
                    payload[key] = image_tools.convert_to_uint8(image_tools.resize_with_pad(np.asarray(Image.open(directory / f"{camera}.png")), 224, 224))
                echoed = client.infer(payload)
                if set(echoed) != set(payload):
                    raise ValueError("Roundtrip schema changed")
                for key, value in payload.items():
                    if isinstance(value, np.ndarray):
                        if value.dtype != echoed[key].dtype or not np.array_equal(value, echoed[key]):
                            raise ValueError("Roundtrip array dtype/pixels/state/noise changed")
                    elif value != echoed[key]:
                        raise ValueError("Roundtrip text changed")
                rows.append({"task": task, "all_arrays_and_prompt_exact": True,
                    "front_sha256": hashlib.sha256(payload["base_rgb"].tobytes()).hexdigest(),
                    "wrist_sha256": hashlib.sha256(payload["wrist_rgb"].tobytes()).hexdigest()})
            client._ws.close()
            atomic_json(args.output / "audit.json", {"passed": True, "rows": rows,
                "model_inference_used": False, "neural_episodes_executed": 0,
                "scope": "actual_Python3.12_simulator_to_Python3.11_openpi_msgpack_websocket_transport_ONLY"})
        finally:
            if server is not None and server.poll() is None:
                server.terminate()
                server.wait(timeout=15)


if __name__ == "__main__":
    main()
