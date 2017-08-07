import matplotlib.pyplot as plt

from method.my_agenda import *

from util.util import *
from util.database_io import *


__author__ = 'Riccardo Guidotti'


def build_myagenda_we(train, tzoffset, reg_loc=True):
    wdwe_train = {
        'wd': {'trajectories': dict()},
        'we': {'trajectories': dict()},
    }

    trajectories = train['trajectories']

    for tid, traj in trajectories.iteritems():
        traj = train['trajectories'][tid]
        ts = datetime.datetime.fromtimestamp(traj.start_point()[2] / 1000 + tzoffset * 3600)
        weekday = ts.weekday()
        wdwe = 'wd' if weekday < 5 else 'we'
        wdwe_train[wdwe]['trajectories'][tid] = traj

    wdwe_model = dict()
    for wdwe in wdwe_train:
        # print weekday, datetime.datetime.now()
        train_wd = wdwe_train[wdwe]
        myagenda_model = build_myagenda(train_wd, tzoffset, reg_loc=reg_loc)
        wdwe_model[wdwe] = myagenda_model

    return wdwe_model


def generate_myagenda_we(wdwe, myagenda_we_model, random_choice=False,
                         time_start=0, time_offset=300, time_stop=86400, starting_lid=None):

    myagenda_model = myagenda_we_model[wdwe]

    if myagenda_model is None:
        return None, None

    agenda, ending_lid = generate_myagenda(myagenda_model, random_choice=random_choice, time_start=time_start,
                                           time_offset=time_offset, time_stop=time_stop, starting_lid=starting_lid)

    return agenda, ending_lid


def generate_myagenda_we_reinforcing(wdwe, myagenda_we_model, real_agenda, reinforcing_time=1800, random_choice=False,
                                     time_start=0, time_offset=300, time_stop=86400, starting_lid=None):

    myagenda_model = myagenda_we_model[wdwe]

    if myagenda_model is None:
        return None, None

    agenda, ending_lid = generate_myagenda_reinforcing(myagenda_model, real_agenda, reinforcing_time,
                                                       random_choice=random_choice, time_start=time_start,
                                                       time_offset=time_offset, time_stop=time_stop,
                                                       starting_lid=starting_lid)

    return agenda, ending_lid


def main():

    params = rome_params

    con = get_connection()
    cur = con.cursor()

    input_table = params['input_table']
    min_lat = params['min_lat']
    min_lon = params['min_lon']
    tzoffset = params['tzoffset']

    uid = 146099 # rome
    #uid = 659447 # london
    week_limit = 42
    weekday = 4
    wdwe = 'wd' if weekday < 5 else 'we'

    imh = load_individual_mobility_history(cur, uid, input_table)

    history_order_dict = get_ordered_history(imh)

    train, test = train_test_partition_weeklimit(imh, history_order_dict, week_limit)

    myagenda_we_model = build_myagenda_we(train, tzoffset)

    myagenda_we, ending_lid = generate_myagenda_we(wdwe, myagenda_we_model, random_choice=False,
                                                   time_start=0, time_offset=300, time_stop=86400, starting_lid=None)

    print '------'

    for event_time in sorted(myagenda_we):
        event = myagenda_we[event_time]
        print datetime.timedelta(seconds=event_time), event[1], event[2], event[3]

if __name__ == "__main__":
    main()
