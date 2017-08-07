from method.homework_agenda import *

from util.util import *
from util.database_io import *


__author__ = 'Riccardo Guidotti'


def build_homework_we(train, history_order_dict, grid_length, time_lenght, min_lon, min_lat, tzoffset):
    wdwe_train = {
        'wd': {'trajectories': dict()},
        'we': {'trajectories': dict()},
    }

    for tid in sorted(history_order_dict, key=history_order_dict.get):
        if tid not in train['trajectories']:
            continue
        traj = train['trajectories'][tid]
        ts = datetime.datetime.fromtimestamp(traj.start_point()[2] / 1000 + tzoffset * 3600)
        weekday = ts.weekday()
        wdwe = 'wd' if weekday < 5 else 'we'
        wdwe_train[wdwe]['trajectories'][tid] = traj

    wdwe_model = dict()
    for wdwe in wdwe_train:
        train_wd = wdwe_train[wdwe]
        homework_model = build_homework(train_wd, history_order_dict, grid_length, time_lenght, min_lon, min_lat,
                                        tzoffset)
        wdwe_model[wdwe] = homework_model

    return wdwe_model


def generate_homework_we_agenda(wdwe, grid_we_model, grid_length, time_lenght, min_lon, min_lat,
                                time_start=0, time_offset=300, time_stop=86400):

    homework_model = grid_we_model[wdwe]

    if homework_model is None:
        return None

    agenda = generate_homework_agenda(homework_model, grid_length, time_lenght, min_lon, min_lat,
                                      time_start=time_start, time_offset=time_offset, time_stop=time_stop)

    return agenda


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

    imh = load_individual_mobility_history(cur, uid, input_table)

    history_order_dict = get_ordered_history(imh)

    train, test = train_test_partition_weeklimit(imh, history_order_dict, week_limit)

    grid_length = 1000
    time_lenght = 3600
    weekday = 6
    wdwe = 'wd' if weekday < 5 else 'we'

    homework_we_model = build_homework_we(train, history_order_dict, grid_length, time_lenght,
                                          min_lon, min_lat, tzoffset)

    homework_we_agenda = generate_homework_we_agenda(wdwe, homework_we_model, grid_length, time_lenght,
                                                     min_lon, min_lat, time_start=0, time_offset=300, time_stop=86400)

    print '------'

    for event_time in sorted(homework_we_agenda):
        event = homework_we_agenda[event_time]
        print datetime.timedelta(seconds=event_time), event[1], event[2], event[3]


if __name__ == "__main__":
    main()
