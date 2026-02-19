import numpy as np
def sample_queryFromdata(data,query_rng,only_test=False):
    """
    This funciton samples a query from all three splits, data.train, data.validation, and data.test.
    """
    if only_test:
        dataSplits=[data.test]
    else:
        dataSplits=[data.train,data.validation,data.test]
    splitId=sample_splitId(dataSplits,query_rng)
    Queryid=sample_Queryid(dataSplits[splitId],query_rng)
    return Queryid,dataSplits[splitId]

def sample_splitId(dataSplits,query_rng):
    """
    This funciton samples a data split from data.train, data.validation, and data.test.
    """
    n_queries =[len(dataSplit.queriesList) for dataSplit in dataSplits]
    total_n_query=sum(n_queries)
    query_ratio = np.array(n_queries)/total_n_query

    splitId = query_rng.choice(len(dataSplits), size=1,
                                     p=query_ratio)
    return  int(splitId)
def sample_Queryid(dataSplit,query_rng):
    """
    This funciton samples a data split from data.train, data.validation, and data.test.
    """
    Queryid = query_rng.choice(dataSplit.queriesList, size=1)[0]
    return  Queryid
def getpositionBias(cutoff,positionBiasSeverity):
    """
    This funciton returns the position bias of each rank.
    """  
    return (1/np.log2(2+np.arange(cutoff)))**positionBiasSeverity
def generateClick(ranking,TrueRel,positionBias,RandomNumberGenerator):
    """
    This funciton generate clicks according to relevance and position bias.
    """      
    RankedRel=TrueRel[ranking]
    rand_var = np.random.rand(len(RankedRel))
    rand_prop = np.random.rand(len(positionBias))
    viewed = rand_prop < positionBias
    clicks = np.logical_and(rand_var < RankedRel, viewed)
    return clicks

# def f_d(time):
#     # return 1
#     return np.exp(-time/10000.0)



def f_d(time, total_steps=10_000, peaks=100):
    """
    Periodic Income Function (tilted square wave).

    Parameters
    ----------
    time : int, float, or array-like
        Timestep(s) to evaluate. 0 ≤ time < total_steps 推荐为整数，但也支持浮点或数组。
    total_steps : int, default 10_000
        Total horizon length. Used to compute the wave period.
    peaks : int, default 100
        Number of peaks across the horizon. Determines the period length.

    Returns
    -------
    ndarray or float
        Income value(s) ∈ [0, 1] at the requested timestep(s).
    """
    time = np.asarray(time, dtype=float)

    period = total_steps / peaks          
    half_period = period / 2              

    pos_in_cycle = np.mod(time, period)

    income = np.where(pos_in_cycle < half_period,
                      pos_in_cycle / half_period,  
                      0.0)                       
    return income
