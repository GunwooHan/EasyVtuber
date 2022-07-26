import time

import torch.nn as nn

import tha2.poser.modes.mode_20
import tha3.poser.modes.standard_float
from torch.nn.functional import interpolate

from args import args

from collections import OrderedDict

class TalkingAnimeLight(nn.Module):
    def __init__(self):
        super(TalkingAnimeLight, self).__init__()
        self.face_morpher = tha2.poser.modes.mode_20.load_face_morpher('pretrained/face_morpher.pt')
        self.two_algo_face_rotator = tha2.poser.modes.mode_20.load_face_rotater('pretrained/two_algo_face_rotator.pt')
        self.combiner = tha2.poser.modes.mode_20.load_combiner('pretrained/combiner.pt')
        self.face_cache = OrderedDict()
        self.tot = 0
        self.hit = 0

    def forward(self, image, mouth_eye_vector, pose_vector, mouth_eye_vector_c, ratio=None):
        x = image.clone()
        if args.perf=='model':
            tic=time.perf_counter()
        input_hash = hash(tuple(mouth_eye_vector_c))
        cached = self.face_cache.get(input_hash)
        self.tot+=1
        if cached is None:
            mouth_eye_morp_image = self.face_morpher(image[:, :, 32:224, 32:224], mouth_eye_vector)
            self.face_cache[input_hash]=mouth_eye_morp_image.detach()
            if len(self.face_cache) > args.max_gpu_cache_len:
                self.face_cache.popitem(last=False)
        else:
            self.hit+=1
            mouth_eye_morp_image=cached
            self.face_cache.move_to_end(input_hash)
        if args.debug and ratio is not None:
            ratio.value=self.hit / self.tot
        if args.perf=='model':
            print(" - face_morpher",(time.perf_counter()-tic)*1000)
            tic=time.perf_counter()
        x[:, :, 32:224, 32:224] = mouth_eye_morp_image
        rotate_image = self.two_algo_face_rotator(x, pose_vector)[:2]
        if args.perf=='model':
            print(" - rotator",(time.perf_counter()-tic)*1000)
            tic=time.perf_counter()
        output_image = self.combiner(rotate_image[0], rotate_image[1], pose_vector)
        if args.perf=='model':
            print(" - combiner",(time.perf_counter()-tic)*1000)
            tic=time.perf_counter()
        return output_image

class TalkingAnime3(nn.Module):
    def __init__(self):
        super(TalkingAnime3, self).__init__()
        self.face_morpher = tha3.poser.modes.standard_float.load_face_morpher('data/models/standard_float/face_morpher.pt')
        self.two_algo_face_body_rotator = tha3.poser.modes.standard_float.load_two_algo_generator('data/models/standard_float/two_algo_face_body_rotator.pt')
        self.editor = tha3.poser.modes.standard_float.load_editor('data/models/standard_float/editor.pt')
        self.face_cache = OrderedDict()
        self.tot = 0
        self.hit = 0

    def forward(self, image, mouth_eye_vector, pose_vector, mouth_eye_vector_c, ratio=None):
        if args.perf=='model':
            tic=time.perf_counter()
        x = image.clone()
        input_hash = hash(tuple(mouth_eye_vector_c))
        cached = self.face_cache.get(input_hash)
        self.tot+=1
        if cached is None:
            mouth_eye_morp_image = self.face_morpher(image[:, :, 32:32 + 192, (32 + 128):(32 + 192 + 128)], mouth_eye_vector)[0]
            self.face_cache[input_hash]=mouth_eye_morp_image.detach()
            if len(self.face_cache) > args.max_gpu_cache_len:
                self.face_cache.popitem(last=False)
        else:
            self.hit+=1
            mouth_eye_morp_image=cached
            self.face_cache.move_to_end(input_hash)
        if args.debug and ratio is not None:
            ratio.value=self.hit / self.tot
        if args.perf=='model':
            print(" - face_morpher",(time.perf_counter()-tic)*1000)
            tic=time.perf_counter()
        x[:, :, 32:32 + 192, (32 + 128):(32 + 192 + 128)] = mouth_eye_morp_image
        x_half=interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        rotate_image = self.two_algo_face_body_rotator(x_half, pose_vector)
        if args.perf=='model':
            print(" - rotator",(time.perf_counter()-tic)*1000)
            tic=time.perf_counter()
        output_image = self.editor(x,
                                   interpolate(rotate_image[1], size=(512, 512), mode='bilinear', align_corners=False),
                                   interpolate(rotate_image[2], size=(512, 512), mode='bilinear', align_corners=False),
                                   pose_vector)[0]
        if args.perf=='model':
            print(" - editor",(time.perf_counter()-tic)*1000)
            tic=time.perf_counter()
        return output_image


class TalkingAnime(nn.Module):
    def __init__(self):
        super(TalkingAnime, self).__init__()

    def forward(self, image, mouth_eye_vector, pose_vector):
        x = image.clone()
        mouth_eye_morp_image = self.face_morpher(image[:, :, 32:224, 32:224], mouth_eye_vector)
        x[:, :, 32:224, 32:224] = mouth_eye_morp_image
        rotate_image = self.two_algo_face_rotator(x, pose_vector)[:2]
        output_image = self.combiner(rotate_image[0], rotate_image[1], pose_vector)
        return output_image