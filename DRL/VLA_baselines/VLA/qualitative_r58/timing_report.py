"""Separate simulation/video/render clocks; do not invent original NN latency."""
from pathlib import Path
from VLA.common.artifact_io import atomic_json
from .common import OUT,read


def main():
    rows=[]
    for item in read(OUT/'produced_pairs.json'):
        pair=Path(item['paired_output'])
        for method in ('Scripted_Oracle','XVLA_terminal_release'):
            folder=pair/method;frames=read(folder/'frames.json');renderer=read(folder/'renderer.json')
            row={'scene':item['scene_id'],'seed':item['seed'],'backup':item['backup'],'method':method,
                'video_frames':len(frames),'video_FPS':30.,'encoded_duration_s':len(frames)/30.,
                'simulation_time_of_first_capture_s':frames[0]['physics_tick']/240.,
                'simulation_time_of_last_capture_s':frames[-1]['physics_tick']/240.,
                'render_wall_seconds':renderer.get('wall_seconds'),
                'original_neural_latency_ms':None,'new_NN_inference_in_video_renderer':False,
                'video_duration_NOT_neural_wall_clock_runtime':True}
            if method=='XVLA_terminal_release':
                proof=read(folder/'replay_proof.json')
                row.update(native_policy_Hz=10.,native_consumed_chunk=30,
                    original_RPC_inputs_verified=proof['actual_original_RPC_inputs_verified'],
                    native_simulation_seconds=proof['benchmark_boundary_physics_tick']/240.,
                    total_captured_physics_seconds=proof['physics_ticks']/240.,
                    original_neural_latency_status='Not recorded in source trace/receipts; no latency estimate inferred from30FPS video')
            else:
                row['original_neural_latency_status']='Not applicable: feedback Oracle, not NN inference'
            rows.append(row)
    atomic_json(OUT/'audit/separate_simulation_video_render_timing.json',{
        'rows':rows,'original_inference_latency_not_measured_in_this_qualitative_replay':True,
        'no_real_time_neural_throughput_comparison_from_video_playback':True})
    print({'timing_rows':len(rows),'missing_original_NN_latency_explicit':True},flush=True)


if __name__=='__main__':main()
