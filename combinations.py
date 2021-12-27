import itertools
num_hazards = 3
num_intervals = 25
num_covariates = 3
index = 0
def covCombinations(num_covariates):
    index = 0
    start = [None]*(2**num_covariates)
    end = [None]*(2**num_covariates)
    for i in itertools.product([0,1], repeat=num_covariates):
        counter = num_covariates
        start_index = 0
        end_index = 0
        print (f"{i}")
        for bit in i:
            if bit == 1:
                while end_index == 0:
                    end_index = counter
                start_index = counter
                counter -= 1
            else:
                counter -= 1
        start[index] = start_index
        end[index] = end_index
        print(f"position of start cov: {start_index}")
        print(f"position of end cov: {end_index}")
        index += 1
    return start, end
start, end = covCombinations(num_covariates)
print(f"Start Array: {start}\nEnd Array: {end}")