import random

num_seeds = 10
seeds = [random.randint(0, 2**32 - 1) for _ in range(num_seeds)]

with open("seeds_list.txt", "w") as f:
    for s in seeds:
        f.write(f"{s}\n")
print("Generated 100 random seeds in seeds_list.txt")
