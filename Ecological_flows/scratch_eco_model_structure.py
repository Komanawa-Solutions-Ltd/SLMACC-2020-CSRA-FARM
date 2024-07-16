"""
created matt_dumont 
on: 7/17/24
"""



def get_expert_weightings(expert_name):
    """
    :param expert_name: str, name of the expert
    :return: dict, weightings for each ecological flow component
    """
    if expert_name == 'expert1':
        return {'thing': 1}
    elif expert_name == 'expert2':
        return {'thing': 2}
    elif expert_name == 'pooled':
        return {'thing': 1.5}
    else:
        raise ValueError(f'invalid expert name {expert_name}')


def run_eco_model(air_temp, r_flow, weightings=None):
    """
    :param air_temp: air temperature in degrees C
    :param r_flow: river flow in m3/s
    :param weightings: dict of weightings for each ecological flow component
    :return: ts_score, annual_scores, detailed_scores
    """

    if weightings is None:
        weightings = {'thing':1} # todo
    else:
        # check weiting strugure
        raise NotImplementedError

    # todo check data structure for air_temp and r_flow
    # todo write some assertions
    # run model

    # apply weightings

    # return output

def run_eco_model_vectorised(datetime, air_temp, r_flow, weightings=None):
    """
    # todo check run time first to see if needed.


    :param datetime: np.ndarray (n,) of datetime objects
    :param air_temp: np.ndarray (n,nsims) of air temperature in degrees C for each simulation,
    :param r_flow: np.ndarray (n,nsims) of river flow in m3/s for each simulation
    :param weightings: dict of weightings for each ecological flow component
    :return: ts_score(nsims), annual_scores(nsims, nyears), detailed_scores(nsims, nyears, ncomponents)
    """

    if weightings is None:
        weightings = {'thing':1} # todo
    else:
        # check weiting strugure
        raise NotImplementedError

    # run model

    # apply weightings

    # return output

if __name__ == '__main__':
    air_temp = None
    r_flow = None
    weightings = get_expert_weightings('expert1')
    out = run_eco_model(air_temp, r_flow, weightings)