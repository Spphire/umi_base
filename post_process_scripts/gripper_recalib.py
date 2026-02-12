import json
import os
import sys
import tarfile
import tempfile

import lz4.frame

from post_process_scripts.utils.BsonReader import load_bson_file, save_bson_dict
from post_process_scripts.utils.Curves import get_dh_curve, get_robotiq_curve
from post_process_scripts.utils.CustomException import NoGripperError, MethodFitError
from post_process_scripts.utils.Optimizor import get_optimal_v2, get_optimal_v1
from post_process_scripts.utils.Visualization import visualize_bson
from post_process_scripts.utils.WidthFromAngle import width_from_angle_v1, width_from_angle_v2



def check_curve_method_calib(bson_path)->bool:
    bson_data = load_bson_file(bson_path)

    if 'gripperAngle' not in bson_data.keys() or 'gripperWidth' not in bson_data.keys():
        raise NoGripperError(
            f"skip data not contains gripper"
        )

    # curve default: robotiq
    best_closed_angle_v2, error_v2 = get_optimal_v2(bson_data['gripperAngle'], bson_data['gripperWidth'], get_robotiq_curve())
    (best_open_angle_v1, best_closed_angle_v1), error_v1 = get_optimal_v1(bson_data['gripperAngle'], bson_data['gripperWidth'], get_robotiq_curve())
    if error_v1>1e-10 and error_v2>1e-10:
        if error_v1>error_v2:
            code = (None, best_closed_angle_v2)
        else:
            code = (best_open_angle_v1, best_closed_angle_v1)
        raise MethodFitError(
            code,
            f"method cant fit: " +\
            f"open&close error={error_v1}, "+\
            f"just close error={error_v2}"
        )
    else:
        if error_v1>error_v2:
            print(f"method: just close error{error_v2}")
            bson_data['closed_angle'] = best_closed_angle_v2
        else:
            print(f"method: open and close error{error_v1}")
            bson_data['closed_angle'] = best_closed_angle_v1
            bson_data['open_angle'] = best_open_angle_v1

    bson_data['curve_type'] = get_dh_curve()
    if 'open_angle' in bson_data.keys():
        # choose open and close method
        bson_data['gripperWidth'] = [width_from_angle_v1(
            ga,
            get_dh_curve(),
            bson_data['open_angle'],
            bson_data['closed_angle']
        ) for ga in bson_data['gripperAngle']]
    else:
        # choose just close method
        bson_data['gripperWidth'] = [width_from_angle_v2(
            ga,
            get_dh_curve(),
            bson_data['closed_angle']
        ) for ga in bson_data['gripperAngle']]

    save_bson_dict(bson_data, bson_path)
    return True


if __name__ == '__main__':
    targz_path = "/mnt/data/shenyibo/workspace/umi_base/.cache/targz_blockq3_1-28/blockq3_1-28.tar.gz"
    output_path = '/mnt/data/shenyibo/workspace/umi_base/.cache/targz_blockq3_1-28/blockq3_1-28_dh'
    with tempfile.TemporaryDirectory(dir="/mnt/data/") as tmpdir:
        try:
            is_lz4 = False
            try:
                with tarfile.open(targz_path, 'r:gz') as tar:
                    tar.extractall(path=tmpdir)
            except tarfile.ReadError:
                # Try lz4 compressed tar
                print("[Download] Trying lz4 decompression...")
                with lz4.frame.open(targz_path, 'rb') as lz4_file:
                    with tarfile.open(fileobj=lz4_file, mode='r|') as tar:
                        tar.extractall(path=tmpdir)
                        is_lz4 = True
            print(f"[Download] Successfully extracted to {tmpdir}")
        except Exception as e:
            raise RuntimeError(f"Failed to extract data: {e}")

        metadata_sessions = {}
        for root, dirs, files in os.walk(tmpdir):
            for file in files:
                if file == 'metadata.json':
                    # json load os.path.join(root, "metadata.json")
                    metadata = json.load(open(os.path.join(root, file), 'r'))
                    if metadata['parent_uuid'] in metadata_sessions.keys():
                        metadata_sessions[metadata['parent_uuid']].append(metadata['uuid'])
                    else:
                        metadata_sessions[metadata['parent_uuid']] = [metadata['uuid']]
        pop_count = 0
        valid_count = 0
        for parent, childs in list(metadata_sessions.items()):
            if len(childs)!=2:
                metadata_sessions.pop(parent)
                pop_count+=1
            else:
                valid_count+=1
        print(f"Found {valid_count} valid sessions, pop {pop_count} invalid sessions")

        valid_sessions = [child for childs in metadata_sessions.values() for child in childs]

        # 遍历 tmpdir 下所有 frame_data.bson 文件
        bson_files = []
        for childs in metadata_sessions.values():
            for child in childs:
                bson_files.append(os.path.join(tmpdir, child, "frame_data.bson"))
        print(f"Found {len(bson_files)} bson files.")

        modify_count=0
        # 执行校准
        for bson_file in bson_files:
            try:
                check_curve_method_calib(bson_file)
                modify_count+=1
                print(f"[Success] Calibrated {bson_file}")

            except NoGripperError as e:
                # 没有 gripper，正常跳过
                print(f"[Skip] {bson_file}: {e}")

            except MethodFitError as e:
                # 方法拟合失败，红色 error
                print(f"\033[31m[Error] {bson_file}: {e}\033[0m")
                visualize_bson(bson_file, e.code[1], e.code[0])
        print(f"Modified {modify_count} bson files.")

        # 重新打包
        if is_lz4:
            output_path += ".tar.gz"
            with lz4.frame.open(output_path, mode='wb') as lz4_out:
                with tarfile.open(fileobj=lz4_out, mode='w') as tar_out:
                    for session in valid_sessions:
                        session_path = os.path.join(tmpdir, session)
                        if os.path.exists(session_path):
                            tar_out.add(session_path, arcname=session)
            print(f"[Output] Saved calibrated data to {output_path}")
        else:
            output_path += ".tar.gz"
            with tarfile.open(output_path, "w:gz") as tar_out:
                for session in valid_sessions:
                    session_path = os.path.join(tmpdir, session)
                    if os.path.exists(session_path):
                        tar_out.add(session_path, arcname=session)
            print(f"[Output] Saved calibrated data to {output_path}")