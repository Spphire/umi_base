#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import zarr
from flask import Flask, Response, abort, jsonify, request

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion_policy.common.space_utils import pose_3d_9d_to_homo_matrix_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Serve an interactive 3D viewer for dualfold replay_buffer.zarr episodes.')
    parser.add_argument('--zarr-path', type=Path, required=True)
    parser.add_argument('--manifest-path', type=Path, default=None)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--max-points', type=int, default=1200)
    parser.add_argument('--axis-stride', type=int, default=30)
    parser.add_argument('--axis-length', type=float, default=0.06)
    return parser.parse_args()


def load_manifest(path: Optional[Path]) -> List[Dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def decimate_indices(count: int, max_points: int) -> np.ndarray:
    if count <= 0:
        return np.empty((0,), dtype=int)
    if max_points <= 0 or count <= max_points:
        return np.arange(count, dtype=int)
    idx = np.unique(np.linspace(0, count - 1, max_points).round().astype(int))
    return idx


def axis_sample_indices(count: int, axis_stride: int) -> np.ndarray:
    if count <= 0:
        return np.empty((0,), dtype=int)
    stride = max(1, axis_stride)
    idx = np.arange(0, count, stride, dtype=int)
    if idx[-1] != count - 1:
        idx = np.append(idx, count - 1)
    return idx


def to_axis_segments(mats: np.ndarray, axis_length: float, axis_stride: int) -> Dict[str, Dict[str, List[Optional[float]]]]:
    result: Dict[str, Dict[str, List[Optional[float]]]] = {}
    if len(mats) == 0 or axis_length <= 0:
        for axis_name in ('x', 'y', 'z'):
            result[axis_name] = {'x': [], 'y': [], 'z': []}
        return result
    indices = axis_sample_indices(len(mats), axis_stride)
    axis_defs = [('x', 0), ('y', 1), ('z', 2)]
    for axis_name, axis_idx in axis_defs:
        xs: List[Optional[float]] = []
        ys: List[Optional[float]] = []
        zs: List[Optional[float]] = []
        for mat in mats[indices]:
            origin = mat[:3, 3]
            end = origin + axis_length * mat[:3, axis_idx]
            xs.extend([float(origin[0]), float(end[0]), None])
            ys.extend([float(origin[1]), float(end[1]), None])
            zs.extend([float(origin[2]), float(end[2]), None])
        result[axis_name] = {'x': xs, 'y': ys, 'z': zs}
    return result


def pose_array_to_payload(arr: np.ndarray, max_points: int, axis_stride: int, axis_length: float) -> Dict[str, Any]:
    mats = pose_3d_9d_to_homo_matrix_batch(arr.astype(np.float64, copy=False))
    idx = decimate_indices(len(mats), max_points)
    mats_ds = mats[idx]
    pos = mats_ds[:, :3, 3] if len(mats_ds) else np.empty((0, 3), dtype=np.float64)
    return {
        'count': int(len(arr)),
        'sample_count': int(len(idx)),
        'xyz': pos.tolist(),
        'axes': to_axis_segments(mats_ds, axis_length=axis_length, axis_stride=axis_stride),
        'start': pos[0].tolist() if len(pos) else None,
        'end': pos[-1].tolist() if len(pos) else None,
    }


def build_app(args: argparse.Namespace) -> Flask:
    zarr_path = args.zarr_path.resolve()
    manifest_path = args.manifest_path.resolve() if args.manifest_path else zarr_path.parent.parent / 'episode_manifest.csv'
    root = zarr.open(str(zarr_path), mode='r')
    episode_ends = np.asarray(root['meta']['episode_ends'][:], dtype=np.int64).reshape(-1)
    manifest_rows = load_manifest(manifest_path)

    app = Flask(__name__)
    app.config['JSON_SORT_KEYS'] = False

    def episode_bounds(episode_idx: int) -> tuple[int, int]:
        if episode_idx < 0 or episode_idx >= len(episode_ends):
            raise IndexError(episode_idx)
        start = int(episode_ends[episode_idx - 1]) if episode_idx > 0 else 0
        end = int(episode_ends[episode_idx])
        return start, end

    def episode_meta(episode_idx: int, start: int, end: int) -> Dict[str, Any]:
        row = manifest_rows[episode_idx] if episode_idx < len(manifest_rows) else {}
        return {
            'episode_idx': episode_idx,
            'label': f"{episode_idx:03d} | {row.get('dataset', 'unknown')} | {row.get('parent_uuid', 'unknown')}",
            'dataset': row.get('dataset', ''),
            'parent_uuid': row.get('parent_uuid', ''),
            'frame_count_manifest': int(row.get('frame_count', end - start)) if row.get('frame_count') else int(end - start),
            'frame_count_zarr': int(end - start),
            'left_mode': row.get('left_mode', ''),
            'right_mode': row.get('right_mode', ''),
        }

    @lru_cache(maxsize=256)
    def load_episode_payload(episode_idx: int) -> Dict[str, Any]:
        start, end = episode_bounds(episode_idx)
        data = root['data']
        payload = {
            'meta': episode_meta(episode_idx, start, end),
            'timestamp': np.asarray(data['timestamp'][start:end]).reshape(-1).astype(np.float64).tolist(),
            'left_robot_tcp_pose': pose_array_to_payload(np.asarray(data['left_robot_tcp_pose'][start:end]), args.max_points, args.axis_stride, args.axis_length),
            'right_robot_tcp_pose': pose_array_to_payload(np.asarray(data['right_robot_tcp_pose'][start:end]), args.max_points, args.axis_stride, args.axis_length),
            'left_eye_tcp_pose': pose_array_to_payload(np.asarray(data['left_eye_tcp_pose'][start:end]), args.max_points, args.axis_stride, args.axis_length) if 'left_eye_tcp_pose' in data else None,
            'right_eye_tcp_pose': pose_array_to_payload(np.asarray(data['right_eye_tcp_pose'][start:end]), args.max_points, args.axis_stride, args.axis_length) if 'right_eye_tcp_pose' in data else None,
        }
        return payload

    @app.get('/healthz')
    def healthz() -> Response:
        return jsonify({
            'ok': True,
            'zarr_path': str(zarr_path),
            'manifest_path': str(manifest_path),
            'episode_count': int(len(episode_ends)),
        })

    @app.get('/api/episodes')
    def api_episodes() -> Response:
        items = []
        for episode_idx in range(len(episode_ends)):
            start, end = episode_bounds(episode_idx)
            items.append(episode_meta(episode_idx, start, end))
        return jsonify({'episodes': items})

    @app.get('/api/episode/<int:episode_idx>')
    def api_episode(episode_idx: int) -> Response:
        try:
            payload = load_episode_payload(episode_idx)
        except IndexError:
            abort(404, f'episode {episode_idx} out of range')
        return Response(json.dumps(payload), mimetype='application/json')

    @app.get('/')
    def index() -> Response:
        html = f'''<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Dualfold Zarr Viewer</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ font-family: sans-serif; margin: 0; background: #f7f6f2; color: #1f2937; }}
    .bar {{ display: flex; gap: 12px; align-items: center; padding: 12px 16px; background: #fff; border-bottom: 1px solid #ddd; position: sticky; top: 0; z-index: 10; }}
    .meta {{ padding: 8px 16px; font-size: 14px; background: #fff9e8; border-bottom: 1px solid #e5d9a5; }}
    #plot {{ width: 100vw; height: calc(100vh - 96px); }}
    select, button, input {{ font-size: 14px; }}
    .spacer {{ flex: 1; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  </style>
</head>
<body>
  <div class="bar">
    <button id="prevBtn">Prev</button>
    <select id="episodeSelect"></select>
    <button id="nextBtn">Next</button>
    <label><input type="checkbox" id="showAxes" checked /> show local axes</label>
    <label><input type="checkbox" id="centerAtHead" /> center at first head point</label>
    <div class="spacer"></div>
    <div class="mono">Y-up viewer | max_points={args.max_points} | axis_stride={args.axis_stride}</div>
  </div>
  <div id="meta" class="meta">loading...</div>
  <div id="plot"></div>
  <script>
    const colors = {{
      left: '#006D77',
      right: '#C44536',
      head: '#222222',
      axisX: '#E63946',
      axisY: '#2A9D8F',
      axisZ: '#457B9D',
      rightEye: '#7C3AED'
    }};

    let episodes = [];
    let currentEpisode = 0;
    let currentPayload = null;

    function shiftedSeries(series, origin) {{
      return series.map(v => (v === null ? null : v - origin));
    }}

    function maybeCenterXYZ(xyz, origin) {{
      if (!origin) return xyz;
      return xyz.map(p => [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]]);
    }}

    function maybeCenterAxes(axes, origin) {{
      if (!origin) return axes;
      const out = {{}};
      for (const axisName of Object.keys(axes)) {{
        out[axisName] = {{
          x: shiftedSeries(axes[axisName].x, origin[0]),
          y: shiftedSeries(axes[axisName].y, origin[1]),
          z: shiftedSeries(axes[axisName].z, origin[2]),
        }};
      }}
      return out;
    }}

    function makeLineTrace(name, xyz, color, width) {{
      return {{
        type: 'scatter3d',
        mode: 'lines',
        name,
        x: xyz.map(p => p[0]),
        y: xyz.map(p => p[1]),
        z: xyz.map(p => p[2]),
        line: {{ color, width }},
      }};
    }}

    function makePointTrace(name, point, color) {{
      if (!point) return null;
      return {{
        type: 'scatter3d',
        mode: 'markers',
        name,
        x: [point[0]],
        y: [point[1]],
        z: [point[2]],
        marker: {{ size: 4, color }},
        showlegend: false,
      }};
    }}

    function makeAxisTraces(prefix, axes) {{
      return [
        {{ type: 'scatter3d', mode: 'lines', name: `${{prefix}}_local_X`, x: axes.x.x, y: axes.x.y, z: axes.x.z, line: {{ color: colors.axisX, width: 3 }}, opacity: 0.85 }},
        {{ type: 'scatter3d', mode: 'lines', name: `${{prefix}}_local_Y`, x: axes.y.x, y: axes.y.y, z: axes.y.z, line: {{ color: colors.axisY, width: 3 }}, opacity: 0.85 }},
        {{ type: 'scatter3d', mode: 'lines', name: `${{prefix}}_local_Z`, x: axes.z.x, y: axes.z.y, z: axes.z.z, line: {{ color: colors.axisZ, width: 3 }}, opacity: 0.85 }},
      ];
    }}

    async function loadEpisodes() {{
      const res = await fetch('/api/episodes');
      const data = await res.json();
      episodes = data.episodes;
      const select = document.getElementById('episodeSelect');
      select.innerHTML = '';
      episodes.forEach(ep => {{
        const opt = document.createElement('option');
        opt.value = ep.episode_idx;
        opt.textContent = ep.label;
        select.appendChild(opt);
      }});
      await loadEpisode(0);
    }}

    async function loadEpisode(idx) {{
      currentEpisode = Math.max(0, Math.min(idx, episodes.length - 1));
      document.getElementById('episodeSelect').value = currentEpisode;
      const res = await fetch(`/api/episode/${{currentEpisode}}`);
      currentPayload = await res.json();
      renderPayload();
    }}

    function renderPayload() {{
      const payload = currentPayload;
      if (!payload) return;
      const center = document.getElementById('centerAtHead').checked;
      const showAxes = document.getElementById('showAxes').checked;
      const head = payload.left_eye_tcp_pose;
      const rightEye = payload.right_eye_tcp_pose;
      const left = payload.left_robot_tcp_pose;
      const right = payload.right_robot_tcp_pose;
      const origin = (center && head && head.start) ? head.start : null;

      const traces = [];
      const headXYZ = head ? maybeCenterXYZ(head.xyz, origin) : [];
      const leftXYZ = maybeCenterXYZ(left.xyz, origin);
      const rightXYZ = maybeCenterXYZ(right.xyz, origin);
      const rightEyeXYZ = rightEye ? maybeCenterXYZ(rightEye.xyz, origin) : [];

      if (head && headXYZ.length) {{
        traces.push(makeLineTrace('head_left_eye', headXYZ, colors.head, 6));
        const headStart = origin ? [0,0,0] : head.start;
        const t = makePointTrace('head_start', headStart, colors.head); if (t) traces.push(t);
        if (showAxes) traces.push(...makeAxisTraces('head_left_eye', maybeCenterAxes(head.axes, origin)));
      }}
      if (rightEye && rightEyeXYZ.length) {{
        traces.push(makeLineTrace('head_right_eye', rightEyeXYZ, colors.rightEye, 4));
        if (showAxes) traces.push(...makeAxisTraces('head_right_eye', maybeCenterAxes(rightEye.axes, origin)));
      }}
      traces.push(makeLineTrace('left_robot_tcp', leftXYZ, colors.left, 5));
      traces.push(makeLineTrace('right_robot_tcp', rightXYZ, colors.right, 5));
      const leftStart = origin ? [left.start[0]-origin[0], left.start[1]-origin[1], left.start[2]-origin[2]] : left.start;
      const rightStart = origin ? [right.start[0]-origin[0], right.start[1]-origin[1], right.start[2]-origin[2]] : right.start;
      const lt = makePointTrace('left_start', leftStart, colors.left); if (lt) traces.push(lt);
      const rt = makePointTrace('right_start', rightStart, colors.right); if (rt) traces.push(rt);
      if (showAxes) {{
        traces.push(...makeAxisTraces('left_robot_tcp', maybeCenterAxes(left.axes, origin)));
        traces.push(...makeAxisTraces('right_robot_tcp', maybeCenterAxes(right.axes, origin)));
      }}

      const meta = payload.meta;
      document.getElementById('meta').innerHTML =
        `<b>episode</b> ${{meta.episode_idx}} | <b>dataset</b> ${{meta.dataset}} | <b>parent</b> <span class="mono">${{meta.parent_uuid}}</span> | ` +
        `<b>frames</b> ${{meta.frame_count_zarr}} | <b>left_mode</b> ${{meta.left_mode || 'n/a'}} | <b>right_mode</b> ${{meta.right_mode || 'n/a'}}`;

      Plotly.newPlot('plot', traces, {{
        title: `Dualfold zarr episode ${{meta.episode_idx}}`,
        margin: {{ l: 0, r: 0, t: 48, b: 0 }},
        legend: {{ x: 0.02, y: 0.98 }},
        scene: {{
          xaxis: {{ title: 'X' }},
          yaxis: {{ title: 'Y (up)' }},
          zaxis: {{ title: 'Z' }},
          aspectmode: 'data',
          dragmode: 'orbit',
          camera: {{ up: {{ x: 0, y: 1, z: 0 }}, eye: {{ x: 1.6, y: 0.9, z: 1.6 }} }},
        }},
      }}, {{ responsive: true }});
    }}

    document.getElementById('episodeSelect').addEventListener('change', (e) => loadEpisode(parseInt(e.target.value, 10)));
    document.getElementById('prevBtn').addEventListener('click', () => loadEpisode(currentEpisode - 1));
    document.getElementById('nextBtn').addEventListener('click', () => loadEpisode(currentEpisode + 1));
    document.getElementById('showAxes').addEventListener('change', renderPayload);
    document.getElementById('centerAtHead').addEventListener('change', renderPayload);

    loadEpisodes();
  </script>
</body>
</html>'''
        return Response(html, mimetype='text/html')

    return app


def main() -> int:
    args = parse_args()
    app = build_app(args)
    app.run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
