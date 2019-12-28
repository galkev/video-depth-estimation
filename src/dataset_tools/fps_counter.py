import time


class FpsCounter(object):
    def __init__(self):
        self.cnt = 0
        self.last = time.time()
        self.smoothing = 0.9
        self.fps_smooth = 30

    def step(self):
        self.cnt += 1
        if (self.cnt % 10) == 0:
            now = time.time()
            dt = now - self.last
            fps = 10 / dt
            self.fps_smooth = (self.fps_smooth * self.smoothing) + (fps * (1.0 - self.smoothing))
            self.last = now

    def __str__(self):
        return str(self.fps_smooth)[:4]
