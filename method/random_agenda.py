import numpy as np

from collections import defaultdict

from util.util import *
from util.database_io import *


__author__ = 'Riccardo Guidotti'


def get_lonlat(loc_cell, grid_length2, min_lon, min_lat):
    lon = loc_cell[0] * grid_length2 + min_lon + grid_length2 / 2
    lat = loc_cell[1] * grid_length2 + min_lat + grid_length2 / 2
    return lon, lat


def build_random(train, history_order_dict, grid_length, min_lon, min_lat):
    grid_length2 = dist2angle(grid_length)

    st_grid = defaultdict(int)

    for tid in sorted(history_order_dict, key=history_order_dict.get):
        if tid not in train['trajectories']:
            continue

        traj = train['trajectories'][tid]

        for ti, xyz in enumerate(traj.object):
            lat = xyz[1]
            lon = xyz[0]

            i = int(np.floor((lon - min_lon) / grid_length2))
            j = int(np.floor((lat - min_lat) / grid_length2))

            st_grid[(i, j)] += 1

    p_move_time = {
        0: 0.014,
        1: 0.008,
        2: 0.005,
        3: 0.003,
        4: 0.004,
        5: 0.010,
        6: 0.023,
        7: 0.053,
        8: 0.062,
        9: 0.060,
        10: 0.059,
        11: 0.061,
        12: 0.063,
        13: 0.058,
        14: 0.054,
        15: 0.059,
        16: 0.071,
        17: 0.076,
        18: 0.075,
        19: 0.066,
        20: 0.043,
        21: 0.029,
        22: 0.023,
        23: 0.019,
    }

    random_model = {
        'st_grid': st_grid,
        'p_move': [0.07, 0.05],
        'n_move': [3.27, 1.65],
        'p_length': [12376.36, 16237.52],
        'p_move_time': p_move_time,
        'avg_speed': 13.88  # m/s
    }

    return random_model


def generate_random_agenda(random_model, grid_length, min_lon, min_lat,
                           time_start=0, time_offset=300, time_stop=86400, starting_cell=None):
    agenda = dict()
    time_clock = time_start
    grid_length2 = dist2angle(grid_length)

    st_grid = random_model['st_grid']
    # p_move = random_model['p_move']
    n_move = random_model['n_move']
    p_length = random_model['p_length']
    p_move_time = random_model['p_move_time']
    avg_speed = random_model['avg_speed']

    if starting_cell is None:
        starting_cell = sorted(st_grid.items(), key=lambda x: x[1], reverse=True)[0][0]

    max_traj = np.ceil(np.random.normal(n_move[0], n_move[1]))
    traj_count = 0

    meters_in_traj = 0
    meters_in_traj_in_cell = 0
    traj_cell_history = dict()
    is_moving = False
    traj_length = None
    next_cell = None

    loc_cell = starting_cell

    lon, lat = get_lonlat(loc_cell, grid_length2, min_lon, min_lat)
    agenda[time_clock] = [time_clock, lat, lon, 'stop']

    # print 'n traj', max_traj

    while time_clock < time_stop:

        time_clock += time_offset

        if is_moving is False and time_clock < time_stop:
            if traj_count < max_traj:
                h = time_clock / 3600
                #print [p_move_time[h], 1.0 - p_move_time[h]]
                move_stop = np.random.choice(['move', 'stop'], 1, p=[p_move_time[h], 1.0 - p_move_time[h]])[0]
                if move_stop == 'move':
                    is_moving = True
                    traj_count += 1
                    traj_length = np.abs(np.random.normal(p_length[0]/2.0, p_length[1]))
                    # print 'length', traj_length
                    meters_traveled = avg_speed * time_offset
                    # print 'meters traveled', meters_traveled
                    meters_in_traj += meters_traveled
                    traj_cell_history[loc_cell] = 0

                    if meters_traveled <= grid_length:
                        meters_in_traj_in_cell += meters_traveled
                        lon, lat = get_lonlat(loc_cell, grid_length2, min_lon, min_lat)
                        agenda[time_clock] = [time_clock, lat, lon, 'move']
                    else:
                        meters_in_traj_in_cell = 0
                        cell_offset = np.random.randint(1, max(2, np.ceil(1.0 * meters_traveled / grid_length)))
                        directions = list()
                        i_options = [loc_cell[0], loc_cell[0] + cell_offset, loc_cell[0] - cell_offset]
                        j_options = [loc_cell[1], loc_cell[1] + cell_offset, loc_cell[1] - cell_offset]

                        for i_opt in i_options:
                            for j_opt in j_options:
                                if (i_opt, j_opt) not in traj_cell_history and (i_opt, j_opt) in st_grid:
                                    directions.append((i_opt, j_opt))

                        if len(directions) == 0:
                            for i_opt in i_options:
                                for j_opt in j_options:
                                    if (i_opt, j_opt) in st_grid:
                                        directions.append((i_opt, j_opt))

                        next_cell = directions[np.random.choice(range(0, len(directions)))]
                        lon, lat = get_lonlat(next_cell, grid_length2, min_lon, min_lat)
                        agenda[time_clock] = [time_clock, lat, lon, 'move']
                    # print 'Leaving location %s at %s' % (loc_cell, time_clock / 3600), datetime.timedelta(seconds=time_clock), is_moving

        if is_moving and time_clock < time_stop:

            if meters_in_traj <= traj_length:
                is_moving = True
                next_cell0 = next_cell
                meters_traveled = avg_speed * time_offset
                meters_in_traj += meters_traveled
                traj_cell_history[next_cell] = 0

                if meters_traveled + meters_in_traj_in_cell <= grid_length:
                    meters_in_traj_in_cell += meters_traveled
                    lon, lat = get_lonlat(loc_cell, grid_length2, min_lon, min_lat)
                    agenda[time_clock] = [time_clock, lat, lon, 'move']
                else:
                    meters_in_traj_in_cell = 0
                    cell_offset = np.random.randint(1, max(2, np.ceil(1.0 * meters_traveled / grid_length)))
                    directions = list()
                    i_options = [loc_cell[0], loc_cell[0] + cell_offset, loc_cell[0] - cell_offset]
                    j_options = [loc_cell[1], loc_cell[1] + cell_offset, loc_cell[1] - cell_offset]

                    for i_opt in i_options:
                        for j_opt in j_options:
                            if (i_opt, j_opt) not in traj_cell_history  and (i_opt, j_opt) in st_grid:
                                directions.append((i_opt, j_opt))

                    if len(directions) == 0:
                        for i_opt in i_options:
                            for j_opt in j_options:
                                if (i_opt, j_opt) in st_grid:
                                    directions.append((i_opt, j_opt))

                    next_cell = directions[np.random.choice(range(0, len(directions)))]
                    lon, lat = get_lonlat(next_cell, grid_length2, min_lon, min_lat)
                    agenda[time_clock] = [time_clock, lat, lon, 'move']
                # assert next_cell0 != next_cell
                # print 'Moving from %s to %s' % (next_cell0, next_cell), datetime.timedelta(seconds=time_clock), is_moving

            else:
                is_moving = False
                loc_cell = next_cell
                meters_in_traj = 0
                meters_in_traj_in_cell = 0
                traj_cell_history = dict()
                traj_length = None
                next_cell = None
                # print 'Arriving in %s' % str(loc_cell), datetime.timedelta(seconds=time_clock), is_moving

        if time_clock < time_stop:
            if not is_moving:
                if loc_cell is None:
                    loc_cell = starting_cell
                lon, lat = get_lonlat(loc_cell, grid_length2, min_lon, min_lat)
                agenda[time_clock] = [time_clock, lat, lon, 'stop']

    ending_cell = loc_cell

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

    random_model = build_random(train, history_order_dict, grid_length, min_lon, min_lat)

    random_agenda = generate_random_agenda(random_model, grid_length, min_lon, min_lat,
                                             time_start=0, time_offset=300, time_stop=86400)

    print '------'

    for event_time in sorted(random_agenda):
        event = random_agenda[event_time]
        print datetime.timedelta(seconds=event_time), event[1], event[2], event[3]

    print '-------'

    moving = False
    for idx, event_time in enumerate(sorted(random_agenda)):
        event = random_agenda[event_time]
        if not moving and event[3] == 'move':
            print '%s,%s' % (random_agenda[sorted(random_agenda)[idx - 1]][1],
                             random_agenda[sorted(random_agenda)[idx - 1]][2])
            moving = True
        if moving and event[3] == 'move':
            print '%s,%s' % (event[1], event[2])
        if moving and event[3] == 'stop':
            print '%s,%s' % (event[1], event[2])
            moving = False
            print '------'

if __name__ == "__main__":
    main()
