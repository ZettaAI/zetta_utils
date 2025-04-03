import random

# Maximum value for _id_nonunique field (2^64 - 1)
MAX_ID_NONUNIQUE = 18446744073709551615  # 2**64 - 1


def generate_nonunique_id() -> int:
    """Generate a random integer ID between 0 and 2^64-1."""
    return random.randint(0, MAX_ID_NONUNIQUE)
