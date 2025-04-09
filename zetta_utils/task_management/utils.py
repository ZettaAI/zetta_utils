import random

MAX_NONUNIQUE_ID = 2 ** 32 - 1


def generate_id_nonunique():
    return random.randint(0, MAX_NONUNIQUE_ID)
