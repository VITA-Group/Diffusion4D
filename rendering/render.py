import os
import subprocess
import concurrent.futures
import subprocess
import logging
import random
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--obj_path", type=str, default="./obj_data/hf-objaverse-v1/glbs")
parser.add_argument("--save_dir", type=str, default='./output')
parser.add_argument("--gpu_num", type=int, default=8)
parser.add_argument("--frame_num", type=int, default=24)
parser.add_argument("--azimuth_aug",  type=int, default=0)
parser.add_argument("--elevation_aug", type=int, default=0,)
parser.add_argument("--resolution", default=256)
parser.add_argument("--mode_multi",  type=int, default=0)
parser.add_argument("--mode_static", type=int, default=0)
parser.add_argument("--mode_front_view",  type=int, default=0)
parser.add_argument("--mode_four_view", type=int, default=0)


args = parser.parse_args()

#read objavsers glb path
glb_files = []
for root, dirs, files in os.walk(args.obj_path):
    for file in files:
        if file.endswith('.glb'):
            glb_file_path = os.path.join(root, file)
            glb_files.append(glb_file_path)


os.makedirs(args.save_dir,exist_ok=True)


def execute_command(args,glb_file, gpu_id,file_name):
    #print('args.azimuth_aug:',args.azimuth_aug)
    if args.azimuth_aug:
        azimuth=round(random.uniform(0, 1), 2)
        file_name+=f'_az{azimuth:.2f}'
    else:
        azimuth=0
    
    if args.elevation_aug:
        elevation= random.randint(5,30)
        file_name+=f'_el{elevation:.2f}'
    else:
        elevation=0
    
    
    save_path=os.path.join(args.save_dir,file_name)
    command=f'CUDA_VISIBLE_DEVICES={gpu_id} export DISPLAY=:0.1 && blender-3.2.2-linux-x64/blender \
        --background --python blender.py -- \
        --object_path {glb_file} \
        --frame_num {args.frame_num} \
        --output_dir {save_path} \
        --gpu_id {gpu_id} \
        --azimuth {azimuth}\
        --elevation {elevation}\
        --resolution {args.resolution} \
        --mode_multi {args.mode_multi}\
        --mode_static {args.mode_static}\
        --mode_front {args.mode_front_view}\
        --mode_four_view {args.mode_four_view}'
                                
    print('command:',command)
    logging.info(f'Executing command: {command}')
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logging.info(result.stdout.decode())
    logging.error(result.stderr.decode())



#distribute glb id to each gpu
tasks = [(glb_file, i % args.gpu_num,os.path.basename(glb_file)) for i, glb_file in enumerate(glb_files)]


#run
with concurrent.futures.ProcessPoolExecutor(max_workers=args.gpu_num) as executor:
    futures = [executor.submit(execute_command, args,glb_file, gpu_id,file_name.split('.')[0]) for glb_file, gpu_id,file_name in tasks]
    logging.basicConfig(level=logging.INFO)
    concurrent.futures.wait(futures)

