import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
parser.add_argument('--extend_movement', type=float)
parser.add_argument('--input', type=str, default='cam')
parser.add_argument('--character', type=str, default='y')
parser.add_argument('--output_dir', type=str)
parser.add_argument('--output_webcam', type=str)
parser.add_argument('--output_size', type=str, default='256x256')
parser.add_argument('--debug_input', action='store_true')
parser.add_argument('--perf', action='store_true')
parser.add_argument('--skip_model', action='store_true')
parser.add_argument('--ifm', type=str)
parser.add_argument('--anime4k', action='store_true')
args = parser.parse_args()
args.output_w = int(args.output_size.split('x')[0])
args.output_h = int(args.output_size.split('x')[1])
if args.output_webcam is None and args.output_dir is None: args.debug = True