"""Actual native collator/labels/RGB/full64 prompt parity, without neural weights."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from VLA.audit_privileged_native_inputs import SNAPSHOT
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.scene64_conditioning import augment_prompt,parse_prompt,PROMPT_SCHEMA
from VLA.scene64_dataset import Scene64NativeDataset


def main():
    import VLA.openvla.train_privileged_prompt as native
    native.PrivilegedNativeDataset=Scene64NativeDataset;native.PROMPT_SCHEMA=PROMPT_SCHEMA
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve each native audit')
    processor=AutoProcessor.from_pretrained(str(SNAPSHOT),trust_remote_code=True,local_files_only=True)
    key=json.loads((a.dataset/'conversion_audit.json').read_text())['dataset_key']
    data=native.NativePromptDataset(a.dataset,processor,ActionTokenizer(processor.tokenizer),PurePromptBuilder,key)
    ids=np.linspace(0,len(data)-1,128,dtype=int).tolist()
    collator=PaddedCollatorForActionPrediction(processor.tokenizer.model_max_length,processor.tokenizer.pad_token_id,padding_side='right')
    count=maximum=0
    for batch in DataLoader(data,batch_size=2,sampler=ids,collate_fn=collator,num_workers=0):
        selected=ids[count:count+2];expected=collator([data[i] for i in selected])
        for field in ('pixel_values','input_ids','labels','attention_mask'):torch.testing.assert_close(batch[field],expected[field],rtol=0,atol=0)
        for i in selected:
            raw=data.raw[i];item=data[i]
            if raw['prompt']!=augment_prompt(raw['original_prompt'],raw['pre_action_scene64']):raise ValueError('Actual64 prompt changed')
            if np.max(np.abs(parse_prompt(raw['prompt'])-raw['pre_action_scene64']))>.000051:raise ValueError('Feedback serialization differs')
            infer=processor.tokenizer('In: What action should the robot take to '+raw['prompt'].lower()+'?\nOut: ',truncation=False).input_ids
            common=0
            for x,y in zip(item['input_ids'].tolist(),infer):
                if x!=y:break
                common+=1
            if common<len(infer)-2 or int((item['labels']!=-100).sum())!=8:raise ValueError('Native action-token suffix or inference prefix mismatch')
            torch.testing.assert_close(item['pixel_values'],processor.image_processor.apply_transform(Image.fromarray(raw['image'])),rtol=0,atol=0)
            maximum=max(maximum,len(item['input_ids']))
        count+=len(batch['input_ids'])
    if count!=128:raise ValueError('Incomplete actual batch test')
    atomic_json(a.output,{'passed':True,'schema':PROMPT_SCHEMA,'actual_loader_batches':64,'samples':count,
        'maximum_actual_training_tokens':maximum,'errors':{'pixel_values':0,'input_ids':0,'labels':0,'attention_mask':0},
        'native_train_inference_image_and_instruction_parity':True,'supervised_suffix_tokens':8,
        'neural_weights_loaded':False,'CUDA_or_model_SR_qualified':False,
        'dataset_contract_sha256':sha256_file(a.dataset/'privileged_dataset.json'),
        'trainer_sha256':sha256_file(Path(__file__).with_name('train_scene64_prompt.py'))})


if __name__=='__main__':main()
