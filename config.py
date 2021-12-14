import math
import numpy as np

batch_size = 20
gamma = 0.95

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 20000


def epsilon_by_frame(request):
    return epsilon_final + (epsilon_start - epsilon_final) * math.exp(
        -1. * request / epsilon_decay)


if __name__ == '__main__':
    print(epsilon_by_frame(150000))