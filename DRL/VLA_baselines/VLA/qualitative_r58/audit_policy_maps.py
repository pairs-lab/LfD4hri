"""Verify the THIRD privileged image against original actual NN receipts.

This reconstructs deterministic input pixels, not actions or new inference.
It also verifies original full consumed chunks against server action hashes
when all30 native actions were recorded. Partial last chunks are explicit.
"""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.privileged_visual_r53.scene_map import render,contract
from .common import OUT,read


def main():
    root=OUT/'audit/policy_inputs';root.mkdir(exist_ok=True)
    results=[];receipts_cache={}
    for item in read(OUT/'assignments.json'):
        cap=Path(item['vla']);proof=read(cap/'replay_proof.json');trace=read(proof['source_trajectory'])['pick_place']
        cohort=Path(proof['source_cohort']);progress=read(cohort/'progress.json')
        receipts_path=Path(progress['actual_neural_query_receipts'])
        if str(receipts_path) not in receipts_cache:
            receipts_cache[str(receipts_path)]={r['input_sha256']:r for line in receipts_path.read_text().splitlines()
                if line.strip() for r in [json.loads(line)]}
        receipts=receipts_cache[str(receipts_path)]
        prefix='backup' if item['backup'] else 'scene'
        directory=root/f'{prefix}{item["scene_id"]:02d}_seed{item["seed"]}';directory.mkdir(exist_ok=True)
        inputs={}
        for role in ('front','wrist'):
            source=cap/'policy_inputs'/('reset_'+role+'.png');target=directory/source.name
            if not target.exists():shutil.copy2(source,target)
            if sha256_file(source)!=sha256_file(target):raise ValueError('Actual policy reset RGB copy differs')
            inputs[role]={'file':str(target.relative_to(OUT)),'sha256':sha256_file(target)}
        queries=[];images=[]
        for start in range(0,len(trace),30):
            control=proof['controls'][start];receipt=receipts[control['neural_chunk_input_sha256']]
            rgb=render(np.asarray(trace[start]['pre_action_scene64'],np.float32),'pick_place')
            digest=hashlib.sha256(rgb.tobytes()).hexdigest()
            if digest!=receipt['privileged_map_RGB_sha256']:
                raise ValueError('Reconstructed third image differs from actual original NN receipt')
            actual=receipt['actual_native_visual_receipt']
            if actual['actual_views']!=3 or not all(all(row) for row in actual['actual_valid_mask']):
                raise ValueError('Original native network did not consume all3 actual image views')
            subset=trace[start:start+30];full=len(subset)==30
            if full:
                actions=np.asarray([r['native_neural_action'] for r in subset],np.float32)
                if hashlib.sha256(actions.tobytes()).hexdigest()!=receipt['action_sha256_float32']:
                    raise ValueError('Full native chunk differs from actual server action receipt')
            image=Image.fromarray(rgb);file=directory/f'query{start:05d}_privileged_map.png';image.save(file);images.append(image)
            queries.append({'control':start,'simulation_time_s':start/10.,'RGB_bytes_SHA256':digest,
                'original_receipt_file':str(receipts_path),'original_receipt_file_SHA256':sha256_file(receipts_path),
                'all3_actual_views_consumed':True,'full30_native_chunk_action_SHA_verified':full,
                'partial_terminal_chunk_not_misrepresented_as_full_chunk':not full,
                'image':str(file.relative_to(OUT))})
        sheet=Image.new('RGB',(384*4,384*((len(images)+3)//4)),(235,238,242))
        for i,image in enumerate(images):sheet.paste(image,(384*(i%4),384*(i//4)))
        sheet.save(directory/'actual_NN_map_queries.png')
        results.append({'scene':item['scene_id'],'backup':item['backup'],'seed':item['seed'],
            'actual_reset_policy_RGB':inputs,'queries':queries,
            'native_chunk_count':len(queries),'NN_processed_tensor_or_rng_recomputation_claimed':False})
    atomic_json(OUT/'audit/third_privileged_visual_input_receipts.json',{'passed':True,'contract':contract(),
        'case_count':len(results),'queries_verified':sum(len(r['queries']) for r in results),
        'all_maps_RGB_bytes_match_original_server_receipts':True,
        'partial_terminal_chunk_hash_NOT_claimed_verified':True,'rows':results})
    pages=[]
    for start in range(0,len(results),3):
        page=Image.new('RGB',(1152,1152),(235,238,242));selected=results[start:start+3]
        for row_index,row in enumerate(selected):
            prefix='backup' if row['backup'] else 'scene'
            directory=root/f'{prefix}{row["scene"]:02d}_seed{row["seed"]}'
            for column,file in enumerate(('reset_front.png','reset_wrist.png','query00000_privileged_map.png')):
                with Image.open(directory/file) as original:page.paste(original.convert('RGB'),(column*384,row_index*384))
        file=root/f'reset_triplets_{start//3+1:02d}.png';page.save(file)
        pages.append({'file':str(file.relative_to(OUT)),
            'rows':[{'scene':r['scene'],'backup':r['backup'],'seed':r['seed']} for r in selected],
            'columns':['actual_front_RGB_at_reset','actual_wrist_RGB_at_reset','receipt_verified_third_GT_image'],
            'no_resizing_or_image_content_repair':True})
    atomic_json(OUT/'audit/reset_triplet_contact_sheet_order.json',pages)
    print({'actual3view_input_cases':len(results),'actual_queries_verified':sum(len(r['queries']) for r in results),
        'all_original_receipt_hashes_passed':True},flush=True)


if __name__=='__main__':main()
