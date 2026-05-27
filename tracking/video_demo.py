import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
print(prj_path)
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import Tracker, tracker


def run_video(tracker_name, tracker_param, videofile, optional_box=None, debug=None, save_results=False):
    """Run the tracker on your webcam.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        debug: Debug level.
    """
    tracker = Tracker(tracker_name, tracker_param, "video")
    tracker.run_video(videofilepath=videofile, optional_box=optional_box, debug=debug, save_results=save_results)
    # Sau vòng lặp track xong

# def main():
#     parser = argparse.ArgumentParser(description='Run the tracker on your webcam.')
#     parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
#     parser.add_argument('tracker_param', type=str, help='Name of parameter file.')
#     parser.add_argument('videofile', type=str, help='path to a video file.')
#     parser.add_argument('--optional_box', type=float, default=None, nargs="+", help='optional_box with format x y w h.')
#     parser.add_argument('--debug', type=int, default=0, help='Debug level.')
#     parser.add_argument('--save_results', dest='save_results', action='store_true', help='Save bounding boxes')
#     parser.set_defaults(save_results=False)

#     args = parser.parse_args()

#     run_video(args.tracker_name, args.tracker_param, args.videofile, args.optional_box, args.debug, args.save_results)
tracker_name = "sglatrack"
tracker_param = "deit_distilled"
videofile = r"/media/getac2/My Passport/tuld3/Uav_tracking/UAV-Anti-UAV/Train/Train/UAV-Anti-UAV_Train_000003/UAV-Anti-UAV_Train_000003.mp4"

parent_dir = os.path.dirname(videofile)
print(parent_dir)

with open(os.path.join(parent_dir, "groundtruth_rect.txt"), "r") as f:
    first_line = f.readline().strip()

print(first_line.split(","))
optional_box = list(map(float, first_line.split(",")))

run_video(tracker_name, tracker_param, videofile, optional_box=optional_box)
# if __name__ == '__main__':
#     run_video(tracker_name, tracker_param, videofile)
