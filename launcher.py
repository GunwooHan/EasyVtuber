import os
import subprocess
import sys
import tkinter as tk
from tkinter import ttk
import json

try:
    f=open('launcher.json')
    args=json.load(f)
    f.close()
except:
    args={
        'character': 'y',
        'input': 2,
        'output': 2,
        'ifm': None,
        'is_extend_movement':False,
        'is_anime4k':False,
    }

p=None

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
    args = {
        'character': character.get(),
        'input': input.get(),
        'output': output.get(),
        'ifm':ifm.get(),
        'is_extend_movement':is_extend_movement.get(),
        'is_anime4k':is_anime4k.get(),
    }
    f=open('launcher.json', mode='w')
    json.dump(args,f)
    if p is not None:
        p.kill()
    p = subprocess.Popen(
        [os.path.join(os.getcwd(),'venv','Scripts','python.exe'),
         'main.py','--debug','--debug_input'])



ttk.Button(launcher, text="Save & Launch", command=launch).pack(fill='x', expand=True, pady=10)

root.mainloop()
