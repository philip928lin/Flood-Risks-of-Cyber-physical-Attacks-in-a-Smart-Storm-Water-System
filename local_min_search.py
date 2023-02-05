import numpy as np

def local_min_search(sim_func, max_iter=10, size=100, bound=[0,1], tol=10e-8,
                     inputs=[]):
    """
    Recursively search over multiple local optimum and return the best.

    Parameters
    ----------
    sim_func : function
        simulation function.
    max_iter : int, optional
        Maximum iteration. The default is 10.
    size : int, optional
        Sample size. The default is 100.
    bound : list, optional
        Sample range. The default is [0,1].
    tol : float, optional
        Convergency tolerance. The default is 10e-8.
    inputs : list, optional
        Inputs for sim_func. Ex sim_func(x, *inputs). The default is [].

    Returns
    -------
    tuple
        sol_x, record.

    """
    def identify_local_optimum(objs):
        opts_index = []
        if objs[0] < objs[1]:
            opts_index.append(0)
        for i in range(1, size-1):
            if objs[i-1] >= objs[i] and objs[i] <= objs[i+1]:
                opts_index.append(i)
        if objs[-1] < objs[-2]:
            opts_index.append(size-1)
        return opts_index
    def get_new_sample_ranges(opts_index, samples):
        ranges = []
        new_opts_index = []
        i_max = len(samples)-1
        for i, index in enumerate(opts_index):
            start_i = max(index-1, 0)
            start = samples[start_i]
            end_i = min(index+1, i_max)
            end = samples[end_i]
            if i != 0:
                if index == opts_index[i-1]+1:
                    ranges[-1][1] = samples[min(index+1, i_max)]
                    new_opts_index[-1][1] = min(index+1, i_max)
                else:
                    ranges.append([start, end])
                    new_opts_index.append([start_i, end_i])
            else:
                ranges.append([start, end])
                new_opts_index.append([start_i, end_i])
        return ranges, new_opts_index
    def create_samples(sample_range, factor=1):
        new_samples = np.linspace(
            sample_range[0], sample_range[1], num=size*factor)
        return new_samples
    def evaluation(samples):
        return [sim_func(value, *inputs) for value in samples]
    def check_convergency(index, objs):          
        diff = (np.abs(objs[index[0]]-objs[index[0]+1]) 
                + np.abs(objs[index[1]-1]-objs[index[1]])) / 2
        if diff < tol:
            print("Reach the tolerance criteria.")
            return True
        else:
            return False
    def local_search(ite, samples, obj=10e8, sol=None, loc_sol=None, 
                     loc_obj=None, converge=False):  
        # Recursively search over multiple local optimum and return the best.        
        if ite == max_iter or converge:           
            print("Local optimum: x= {}, obj= {}".format(loc_sol, loc_obj))
            if loc_obj < obj:
                obj = loc_obj
                sol = loc_sol
            return sol, obj
        else:
            print("{}:".format(ite))
            objs = evaluation(samples)
            opts_index = identify_local_optimum(objs)
            ranges, opts_index = get_new_sample_ranges(opts_index, samples)
            if len(ranges) > 1:
                print("Found {} of searching regions.".format(len(ranges)))
                print(ranges)
            # print("opt_sols: ", [samples[i] for i in opts_index])
            # print("opt_objs: ", [objs[i] for i in opts_index])
            for opt_index, sample_range in zip(opts_index, ranges):
                new_samples = create_samples(sample_range)
                center_index = int(np.mean(opt_index))
                loc_sol, loc_obj = (samples[center_index], objs[center_index])
                sol, obj = local_search(
                    ite+1, new_samples, obj, sol, loc_sol, loc_obj,
                    check_convergency(opt_index, objs))
        return sol, obj
    sol, obj = local_search(ite=0, samples=create_samples(bound))
    print("Best: x= {}, obj= {}".format(sol, obj))
    return sol, obj

r"""
def test_func1(x):
    return min( np.abs(x-0.3792414)-0.001, np.abs(x-0.8432414))
def test_func2(x):
    return min( np.abs(x-0.3792414), np.abs(x-0.8432414)-0.001)
def test_func3(x):
    y=0.5
    if x<0.34:
        y=0
    return y

def test_func4(x):
    y=0.5
    if x >= 0.34 and x <= 0.56:
        y=0
    return y
#%%
x_sol, obj = local_min_search(test_func1, max_iter=10, size=100,
                                    bound=[0,1], tol=10e-10)
print(x_sol)
#%%
x_sol, obj = local_min_search(test_func2, max_iter=10, size=100,
                                    bound=[0,1], tol=10e-10)
print(x_sol)
#%%
x_sol, obj = local_min_search(test_func3, max_iter=10, size=100,
                                    bound=[0,1], tol=10e-10)
print(x_sol)
#%%
x_sol, obj = local_min_search(test_func4, max_iter=10, size=100,
                                    bound=[0,1], tol=10e-10)
print(x_sol)
"""
