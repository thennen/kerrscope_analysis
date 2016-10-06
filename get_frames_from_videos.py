# Movie files with large size, don't want to copy them
# want to take some frames out of them and put them somewhere else

import os

pjoin = os.path.join
psplit = os.path.split
splitext = os.path.splitext

fps_out = 1

source_dir = r'\\132.239.170.55\Users\User\Desktop\Tolley\Tolley Os2A Skyr Vids\Day 3 Formation Events'

target_dir = r'\\132.239.170.55\SharableDMIsamples\circle_counting'

vid_fns = [fn for fn in os.listdir(source_dir) if fn.endswith('.avi')]

for v in vid_fns:
    frame_folder = pjoin(target_dir, splitext(v)[0])
    if not os.path.isdir(frame_folder):
        os.makedirs(frame_folder)
    vid_path = pjoin(source_dir, v)
    cmd = 'ffmpeg -i \"{}\" -r {} \"{}\\frame_%04d.png\"'.format(vid_path, fps_out, frame_folder)
    os.system(cmd)
