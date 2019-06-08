import numpy as np


class ReplayBuffer:

    def __init__(self, s_dim, a_dim, buff_size):
        self.s_buf = np.zeros([buff_size, s_dim], dtype=np.float32)
        self.s2_buf = np.zeros([buff_size, s_dim], dtype=np.float32)
        self.a_buf = np.zeros([buff_size, a_dim], dtype=np.float32)
        self.r_buf = np.zeros(buff_size, dtype=np.float32)
        self.done_buf = np.zeros(buff_size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, buff_size

    def store(self, s, a, r, s2, done):
        self.s_buf[self.ptr] = s
        self.s2_buf[self.ptr] = s2
        self.a_buf[self.ptr] = a
        self.r_buf[self.ptr] = r
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(s=self.s_buf[idxs], s2=self.s2_buf[idxs], a=self.a_buf[idxs], r=self.r_buf[idxs], done=self.done_buf[idxs])

