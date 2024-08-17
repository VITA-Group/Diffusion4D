import argparse
import objaverse
import multiprocessing

parser = argparse.ArgumentParser()
parser.add_argument("--id_path", type=str, default="src/sample.txt")
args = parser.parse_args()
multiprocessing_cpu_count=multiprocessing.cpu_count()

with open(args.id_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()
object_uids = [line.strip() for line in lines]
# print(object_uids)
print('file number:', len(object_uids))

# objaverse.BASE_PATH = '/home/projects/Diffusion4D/rendering/'
objaverse._VERSIONED_PATH = 'obj_v1/'
objaverse.load_objects(
    uids=object_uids,
    download_processes=multiprocessing_cpu_count
)
print('download finished')
