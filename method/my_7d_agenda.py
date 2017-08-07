import copy

import matplotlib.pyplot as plt

from method.my_agenda import *

from util.util import *
from util.database_io import *


__author__ = 'Riccardo Guidotti'


def build_myagenda_7d(train, tzoffset, reg_loc=True):
    weekday_train = {
        0: {'trajectories': dict()},
        1: {'trajectories': dict()},
        2: {'trajectories': dict()},
        3: {'trajectories': dict()},
        4: {'trajectories': dict()},
        5: {'trajectories': dict()},
        6: {'trajectories': dict()},
    }

    trajectories = train['trajectories']

    for tid, traj in trajectories.iteritems():
        traj = train['trajectories'][tid]
        ts = datetime.datetime.fromtimestamp(traj.start_point()[2] / 1000 + tzoffset * 3600)
        weekday = ts.weekday()
        weekday_train[weekday]['trajectories'][tid] = traj

    weekday_model = dict()
    for weekday in weekday_train:
        # print weekday, datetime.datetime.now()
        train_wd = weekday_train[weekday]
        myagenda_model = build_myagenda(train_wd, tzoffset, reg_loc=reg_loc)
        weekday_model[weekday] = myagenda_model

    return weekday_model


def generate_myagenda_7d(weekday, myagenda_7d_model, random_choice=False,
                         time_start=0, time_offset=300, time_stop=86400, starting_lid=None):

    myagenda_model = myagenda_7d_model[weekday]
    if myagenda_model is None:
        if len(myagenda_7d_model) > 0:
            closest_weekday = np.min([abs(x - weekday) for x in myagenda_7d_model]) + weekday
            if closest_weekday in myagenda_7d_model:
                myagenda_model = myagenda_7d_model[closest_weekday]
            else:
                myagenda_model = myagenda_7d_model[myagenda_7d_model.keys()[0]]

    if myagenda_model is None:
        return None, None

    agenda, ending_lid = generate_myagenda(myagenda_model, random_choice=random_choice, time_start=time_start,
                                            time_offset=time_offset, time_stop=time_stop, starting_lid=starting_lid)

    return agenda, ending_lid


def generate_myagenda_7d_reinforcing(weekday, myagenda_7d_model, real_agenda, reinforcing_time=1800,
                                     random_choice=False, time_start=0, time_offset=300, time_stop=86400,
                                     starting_lid=None):

    myagenda_model = myagenda_7d_model[weekday]
    if myagenda_model is None:
        if len(myagenda_7d_model) > 0:
            closest_weekday = np.min([abs(x - weekday) for x in myagenda_7d_model]) + weekday
            if closest_weekday in myagenda_7d_model:
                myagenda_model = myagenda_7d_model[closest_weekday]
            else:
                myagenda_model = myagenda_7d_model[myagenda_7d_model.keys()[0]]

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
    weekday = 5

    imh = load_individual_mobility_history(cur, uid, input_table)

    history_order_dict = get_ordered_history(imh)

    train, test = train_test_partition_weeklimit(imh, history_order_dict, week_limit)

    myagenda_7d_model = build_myagenda_7d(train, tzoffset)

    myagenda_7d, ending_lid = generate_myagenda_7d(weekday, myagenda_7d_model, random_choice=False,
                                                   time_start=0, time_offset=300, time_stop=86400, starting_lid=None)

    print '------'

    for event_time in sorted(myagenda_7d):
        event = myagenda_7d[event_time]
        print datetime.timedelta(seconds=event_time), event[1], event[2], event[3]

if __name__ == "__main__":
    main()
