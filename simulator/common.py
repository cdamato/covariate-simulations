import math

models = ["GM", "NB2", "DW2", "DW3", "S", "IFRSB", "IFRGSB"]

def hazard_numerical(model, i, args):
    if model == "GM":
        f = args[0]
    elif model == "DW3":
        f = 1 - math.exp(-args[0] * i**args[1])
    elif model == "DW2":
        f = 1 - args[0]**(i**2 - (i - 1)**2)
    elif model == "IFRGSB":
        f = 1 - args[0] / ((i - 1) * args[1] + 1)
    elif model == "IFRSB":
        f = 1 - args[0] / i
    elif model == "NB2":
        f = (i * args[0]**2)/(1 + args[0] * (i - 1))
    elif model == "S":
        f = args[0] * (1 - args[1]**i)
    elif model == "TL":
        try:
            f = (1 - math.exp(-1/args[1])) / (1 + math.exp(- (i - args[0])/args[1]))
        except OverflowError:
            f = float('inf')
    return f

def g(x, n, betas, interval): # Equation 3
	g = 0
	for i in range(0, n):
		g += betas[i] * x[i][interval]
	g = math.exp(g)
	return g

def p(model, params, interval, x, n, beta): # Equation 2
	pixi = 1 - pow(1 - hazard_numerical(model, interval + 1, params), g(x, n, beta, interval))
	for k in range(0, interval):
		pixi *= pow(1 - hazard_numerical(model, k + 1, params), g(x, n, beta, k))
	return pixi