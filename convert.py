# -*- encoding: utf-8 -*-

import os
from pathlib import Path
input_dir = Path(r"F:\test\cleaned")
output_dir = Path(r"F:\test\samp")
out_audio_file = output_dir / 'audio_paths'
out_text = output_dir / 'text'

idx = 1
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.endswith('.mp3'):
            with open(out_audio_file, 'a',encoding='utf-8') as f:
                f.write(f'hug_id_{idx:04d}\t{Path(root)/file}\n')
            txt_file = file.replace('.mp3', '.txt')
            content = open(Path(root) / txt_file, 'r',encoding='utf-8').read()
            with open(out_text, 'a',encoding='utf-8') as f:
                f.write(f'hug_id_{idx:04d}\t{content}\n')
            idx += 1
        
                

