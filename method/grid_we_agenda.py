from method.grid_agenda import *

from util.util import *
from util.database_io import *

__author__ = 'Riccardo Guidotti'


def build_grid_we(train, history_order_dict, grid_length, time_lenght, min_lon, min_lat, tzoffset):

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
        grid_model = build_st_grid(train_wd, history_order_dict, grid_length, time_lenght, min_lon, min_lat, tzoffset)
        wdwe_model[wdwe] = grid_model

    return wdwe_model


def generate_grid_we_agenda(wdwe, grid_we_model, grid_length, time_lenght, min_lon, min_lat, tzoffset,
                            random_choice=False, time_start=0, time_offset=300, time_stop=86400, starting_cell=None):

    grid_model = grid_we_model[wdwe]

    if grid_model is None:
        return None, None

    agenda, ending_cell = generate_st_grid_agenda(grid_model, grid_length, time_lenght, min_lon, min_lat, tzoffset,
                                                  random_choice=random_choice, time_start=time_start,
                                                  time_offset=time_offset, time_stop=time_stop,
                                                  starting_cell=starting_cell)

    return agenda, ending_cell


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
    weekday = 5
    wdwe = 'wd' if weekday < 5 else 'we'

    grid_we_model = build_grid_we(train, history_order_dict, grid_length, time_lenght, min_lon, min_lat, tzoffset)

    grid_we_agenda, ending_cell = generate_grid_we_agenda(wdwe, grid_we_model, grid_length, time_lenght,
                                                          min_lon, min_lat, tzoffset, random_choice=False,
                                                          time_start=0, time_offset=300, time_stop=86400)

    print '------'

    for event_time in sorted(grid_we_agenda):
        event = grid_we_agenda[event_time]
        print datetime.timedelta(seconds=event_time), event[1], event[2], event[3]

if __name__ == "__main__":
    main()
