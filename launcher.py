import os
import signal
import subprocess
import sys
import tkinter as tk
from tkinter import ttk
import json

try:
    f = open('launcher.json')
    args = json.load(f)
    f.close()
except:
    args = {
        'character': 'y',
        'input': 2,
        'output': 2,
        'ifm': None,
        'is_extend_movement': False,
        'is_anime4k': False,
    }

p = None

root = tk.Tk()
root.resizable(False, False)
root.title('EasyVtuber Launcher')

launcher = ttk.Frame(root)
launcher.pack(padx=10, pady=10, fill='x', expand=True)

character = tk.StringVar(value=args['character'])
ttk.Label(launcher, text="Character").pack(fill='x', expand=True)

ttk.Entry(launcher, textvariable=character).pack(fill='x', expand=True)

input = tk.IntVar(value=args['input'])
ttk.Label(launcher, text="Face Data Source").pack(fill='x', expand=True)
ttk.Radiobutton(launcher, text='iFacialMocap', value=0, variable=input).pack(fill='x', expand=True)
ttk.Radiobutton(launcher, text='Webcam', value=1, variable=input).pack(fill='x', expand=True)
ttk.Radiobutton(launcher, text='Initial Debug Input', value=2, variable=input).pack(fill='x', expand=True)

ttk.Label(launcher, text="iFacialMocap IP:Port").pack(fill='x', expand=True)

ifm = tk.StringVar(value=args['ifm'])
ttk.Entry(launcher, textvariable=ifm, state=False).pack(fill='x', expand=True)

ttk.Label(launcher, text="Extra Options").pack(fill='x', expand=True)
is_extend_movement = tk.BooleanVar(value=args['is_extend_movement'])
ttk.Checkbutton(launcher, text='Extend Movement', variable=is_extend_movement).pack(fill='x', expand=True)

is_anime4k = tk.BooleanVar(value=args['is_anime4k'])
ttk.Checkbutton(launcher, text='Anime4K', variable=is_anime4k).pack(fill='x', expand=True)

output = tk.IntVar(value=args['output'])
ttk.Label(launcher, text="Output").pack(fill='x', expand=True)
ttk.Radiobutton(launcher, text='Unity Capture', value=0, variable=output).pack(fill='x', expand=True)
ttk.Radiobutton(launcher, text='OBS Virtual Camera', value=1, variable=output).pack(fill='x', expand=True)
ttk.Radiobutton(launcher, text='Initial Debug Output', value=2, variable=output).pack(fill='x', expand=True)


def launch():
    global p
    global launch_btn
    args = {
        'character': character.get(),
        'input': input.get(),
        'output': output.get(),
        'ifm': ifm.get(),
        'is_extend_movement': is_extend_movement.get(),
        'is_anime4k': is_anime4k.get(),
    }
    f = open('launcher.json', mode='w')
    json.dump(args, f)
    f.close()
    if p is not None:
        subprocess.run(['taskkill','/F','/PID',str(p.pid),'/T'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        p=None
        launch_btn.config(text="Save & Launch")
    else:
        run_args = [os.path.join(os.getcwd(), 'venv', 'Scripts', 'python.exe'), 'main.py']
        if len(args['character']):
            run_args.append('--character')
            run_args.append(args['character'])

        if args['input'] == 0:
            if len(args['ifm']):
                run_args.append('--ifm')
                run_args.append(args['ifm'])
        elif args['input'] == 1:
            run_args.append('--input')
            run_args.append('cam')
        elif args['input'] == 2:
            run_args.append('--debug_input')

        if args['output'] == 0:
            run_args.append('--output_webcam')
            run_args.append('unitycapture')
        elif args['input'] == 1:
            run_args.append('--output_webcam')
            run_args.append('obs')
        elif args['input'] == 2:
            run_args.append('--debug')
        if args['is_anime4k']:
            run_args.append('--anime4k')
        if args['is_extend_movement']:
            run_args.append('--extend_movement')
            run_args.append('1')
            run_args.append('--output_size')
            run_args.append('512x512')

        p = subprocess.Popen(run_args)
        launch_btn.config(text='Stop')


launch_btn = ttk.Button(launcher, text="Save & Launch", command=launch)
launch_btn.pack(fill='x', expand=True, pady=10)

def closeWindow():
    if p is not None:
        subprocess.run(['taskkill','/F','/PID',str(p.pid),'/T'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    root.destroy()


root.protocol('WM_DELETE_WINDOW', closeWindow)
root.mainloop()
