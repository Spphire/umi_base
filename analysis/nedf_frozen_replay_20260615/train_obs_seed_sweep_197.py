#!/usr/bin/env python3
import argparse, json, os, random, sys
from pathlib import Path
import numpy as np
import torch
import zarr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_frozen_replay_197 import load_policy, post_process_action, quat_angle_metrics, step_metrics
from diffusion_policy.common.action_utils import relative_actions_to_absolute_actions
from diffusion_policy.common.pytorch_util import dict_apply


def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def episode_ranges(ends):
    s=0
    for i,e in enumerate(ends):
        yield i,s,int(e)
        s=int(e)


def choose_obs_indices(root, count, seed, min_remaining=30):
    ends=root['meta/episode_ends'][:].astype(int)
    candidates=[]
    for epi,s,e in episode_ranges(ends):
        # Need 2 obs frames, base at idx+1. Prefer enough future action horizon.
        lo=s
        hi=e-2
        if hi < lo: continue
        # keep all valid; annotate remaining for later
        for idx in range(lo, hi+1):
            candidates.append((epi, idx, e-idx-1))
    rng=np.random.default_rng(seed)
    if count >= len(candidates):
        return candidates
    # stratify: half with enough future, half from all, so short-tail padding cases are represented.
    enough=[c for c in candidates if c[2] >= min_remaining]
    chosen=[]
    n1=min(len(enough), count//2)
    if n1:
        chosen.extend([enough[i] for i in rng.choice(len(enough), n1, replace=False)])
    remaining=count-len(chosen)
    pool=candidates
    chosen_set={(c[0],c[1]) for c in chosen}
    pool=[c for c in pool if (c[0],c[1]) not in chosen_set]
    if remaining:
        chosen.extend([pool[i] for i in rng.choice(len(pool), remaining, replace=False)])
    chosen.sort(key=lambda x:x[1])
    return chosen


def make_obs(root, idx):
    obs={}
    for key in ['left_wrist_img','right_wrist_img']:
        imgs=root['data/'+key][idx:idx+2]
        # zarr already 224x224 uint8, training dataset uses T,H,W,C -> T,C,H,W /255
        obs[key]=(imgs.astype(np.float32)/255.0).transpose(0,3,1,2)
    left=root['data/left_robot_tcp_pose'][idx:idx+2,:9].astype(np.float32)
    right=root['data/right_robot_tcp_pose'][idx:idx+2,:9].astype(np.float32)
    base_abs=np.concatenate([left[-1], right[-1]], axis=0)
    return obs, base_abs


def run_one(policy, obs_tensor, base_abs, action_representation, seed):
    set_all_seeds(seed)
    with torch.no_grad():
        result=policy.predict_action(obs_tensor)
    raw=result.get('action', result.get('action_pred'))[0].detach().cpu().numpy()
    abs_action=relative_actions_to_absolute_actions(raw, base_abs, action_representation)
    final=post_process_action(abs_action)
    q=quat_angle_metrics(final)
    return {
        'seed': int(seed),
        'raw_max_step_l2': step_metrics(raw)['max_step_l2'],
        'final_max_step_l2': step_metrics(final)['max_step_l2'],
        'left_max_deg': q['left']['max_angle_deg'],
        'right_max_deg': q['right']['max_angle_deg'],
        'max_deg': max(q['left']['max_angle_deg'], q['right']['max_angle_deg']),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--zarr', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--num-inference-steps', type=int, default=16)
    ap.add_argument('--obs-count', type=int, default=100)
    ap.add_argument('--seed-count', type=int, default=100)
    ap.add_argument('--obs-seed', type=int, default=42)
    ap.add_argument('--obs-offset', type=int, default=0)
    ap.add_argument('--seed-start', type=int, default=0)
    ap.add_argument('--threshold-deg', type=float, default=90.0)
    args=ap.parse_args()
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    os.environ.setdefault('HF_HUB_OFFLINE','1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root=zarr.open(args.zarr, mode='r')
    obs_indices_all=choose_obs_indices(root,args.obs_offset + args.obs_count,args.obs_seed)
    obs_indices=obs_indices_all[args.obs_offset:args.obs_offset + args.obs_count]
    policy, action_representation, cfg=load_policy(Path(args.ckpt), args.num_inference_steps, device)
    rows=[]; bad=[]
    seeds=list(range(args.seed_start,args.seed_start+args.seed_count))
    for obs_i,(epi,idx,remaining) in enumerate(obs_indices):
        model_obs, base_abs=make_obs(root, idx)
        obs_tensor=dict_apply(model_obs, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
        obs_max=0.0; obs_bad=0; obs_top=None
        for seed in seeds:
            r=run_one(policy, obs_tensor, base_abs, action_representation, seed)
            row={'obs_i':obs_i,'episode':int(epi),'frame_idx':int(idx),'remaining_after_base':int(remaining), **r}
            rows.append(row)
            if r['max_deg'] > obs_max:
                obs_max=r['max_deg']; obs_top=row
            if r['max_deg'] >= args.threshold_deg:
                obs_bad += 1; bad.append(row)
        print(f"{args.name} obs={obs_i}/{len(obs_indices)} epi={epi} idx={idx} remain={remaining} max_deg={obs_max:.2f} bad={obs_bad}/{len(seeds)} top_seed={obs_top['seed'] if obs_top else None}", flush=True)
    result={
        'name':args.name,
        'zarr':args.zarr,
        'ckpt':args.ckpt,
        'num_inference_steps':args.num_inference_steps,
        'obs_count':len(obs_indices),
        'seed_count':len(seeds),
        'threshold_deg':args.threshold_deg,
        'task_name':str(cfg.task.name),
        'action_representation':action_representation,
        'obs_indices':[{'obs_i':i,'episode':int(e),'frame_idx':int(idx),'remaining_after_base':int(rem)} for i,(e,idx,rem) in enumerate(obs_indices)],
        'rows':rows,
        'bad_rows':bad,
    }
    out=Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result,indent=2,ensure_ascii=True),encoding='utf-8')
    print('wrote', out, 'bad', len(bad), 'total', len(rows), flush=True)

if __name__=='__main__':
    main()
