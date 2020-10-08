import pandas as pd
from surprise.model_selection import cross_validate

# compare 3 algorithms using given data
def RecSysCompare(data, algo1, algo2, algo3):
    benchmark = []
    for algorithm in [algo1, algo2, algo3]:
        print('Testing' + str(algorithm))
        results = cross_validate(algorithm, data, measures=['RMSE'], cv=5, verbose = False)
        print('Cross Validation Complete')
        temp = pd.DataFrame.from_dict(results).mean(axis=0)
        print('Averaging Mean')
        temp = temp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
        print('Noting the results')
        benchmark.append(temp)
        print('Testing complete')
    
    surprise_results = pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')
    return surprise_results