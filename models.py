import time

import torch.nn as nn

from tha2.poser.modes.mode_20 import load_face_morpher, load_face_rotater, load_combiner

from args import args

from collections import OrderedDict

class TalkingAnimeLight(nn.Module):
    def __init__(self):
        super(TalkingAnimeLight, self).__init__()
        self.face_morpher = load_face_morpher('pretrained/face_morpher.pt')
        self.two_algo_face_rotator = load_face_rotater('pretrained/two_algo_face_rotator.pt')
        self.combiner = load_combiner('pretrained/combiner.pt')
        self.face_cache = OrderedDict()
        self.tot = 0
        self.hit = 0

    def forward(self, image, mouth_eye_vector, pose_vector, mouth_eye_vector_c, ratio=None):
        x = image.clone()
        if args.perf:
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
        if args.perf:
            print(" - face_morpher",(time.perf_counter()-tic)*1000)
            tic=time.perf_counter()
        x[:, :, 32:224, 32:224] = mouth_eye_morp_image
        rotate_image = self.two_algo_face_rotator(x, pose_vector)[:2]
        if args.perf:
            print(" - rotator",(time.perf_counter()-tic)*1000)
            tic=time.perf_counter()
        output_image = self.combiner(rotate_image[0], rotate_image[1], pose_vector)
        if args.perf:
            print(" - combiner",(time.perf_counter()-tic)*1000)
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