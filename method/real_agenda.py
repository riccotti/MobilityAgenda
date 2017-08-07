from util.util import *
from util.database_io import *


__author__ = 'Riccardo Guidotti'


def build_real_model(imh, history_order_dict, day):

    # days = list()
    # for tid in sorted(history_order_dict, key=history_order_dict.get):
    #     if tid not in imh['trajectories']:
    #         continue
    #     traj = imh['trajectories'][tid]
    #     ts = datetime.datetime.fromtimestamp(traj.start_point()[2] / 1000.0)
    #     days.append(ts.timetuple().tm_yday)
    #
    # day = np.random.choice(days)

    real_model = dict()
    for tid in sorted(history_order_dict, key=history_order_dict.get):
        if tid not in imh['trajectories']:
            continue
        traj = imh['trajectories'][tid]
        ts = datetime.datetime.fromtimestamp(traj.start_point()[2] / 1000.0)
        if ts.timetuple().tm_yday == day:
            real_model[tid] = traj

    return real_model


def generate_real_agenda(real_model, history_order_dict, tzoffset,
                         time_start=0, time_offset=300, time_stop=86400):

    agenda = dict()
    time_clock = time_start
    is_moving = False

    traj_idx = 0
    idx_in_traj = 0
    n_times_between_points = 0
    between_points_idxs = None

    next_tid = sorted(real_model, key=history_order_dict.get)[traj_idx]
    starting_loc = real_model[next_tid].start_point()

    cur_loc = starting_loc

    lon = cur_loc[0]
    lat = cur_loc[1]

    leaving_time = (cur_loc[2] / 1000) % 86400 + tzoffset * 3600
    arrival_time = (real_model[next_tid].end_point()[2] / 1000) % 86400 + tzoffset * 3600

    agenda[time_clock] = [time_clock, lat, lon, 'stop']
    while time_clock < time_stop:

        time_clock += time_offset

        if is_moving is False and time_clock - time_offset <= leaving_time <= time_clock:
            is_moving = True
            cur_tid = next_tid
            cur_time_in_traj = (real_model[cur_tid].point_n(idx_in_traj)[2] / 1000) % 86400 + tzoffset * 3600

            while cur_time_in_traj < time_clock and idx_in_traj + 1 < len(real_model[cur_tid]):
                idx_in_traj += 1
                cur_time_in_traj = (real_model[cur_tid].point_n(idx_in_traj)[2] / 1000) % 86400 + tzoffset * 3600

            a = real_model[cur_tid].point_n(idx_in_traj - 1)
            b = real_model[cur_tid].point_n(idx_in_traj)

            if between_points_idxs is None or between_points_idxs < idx_in_traj - 1:
                between_points_idxs = idx_in_traj - 1
                n_times_between_points = 1
            else:
                n_times_between_points += 1

            cur_point_in_traj = point_at_time(a, b, n_times_between_points * time_offset)
            # print 'Leaving location %s at %s' % (cur_loc[:2], (cur_loc[2] / 1000) % 86400 + tzoffset * 3600), \
            #     datetime.timedelta(seconds=time_clock), is_moving

        if is_moving and time_clock < arrival_time:
            is_moving = True
            cur_time_in_traj = (real_model[cur_tid].point_n(idx_in_traj)[2] / 1000) % 86400 + tzoffset * 3600
            while cur_time_in_traj < time_clock:
                idx_in_traj += 1
                cur_time_in_traj = (real_model[cur_tid].point_n(idx_in_traj)[2] / 1000) % 86400 + tzoffset * 3600

            a = real_model[cur_tid].point_n(idx_in_traj - 1)
            b = real_model[cur_tid].point_n(idx_in_traj)
            # print ''
            # print idx_in_traj - 1, idx_in_traj, time_clock, (a[2] / 1000) % 86400 + tzoffset * 3600, (b[2] / 1000) % 86400 + tzoffset * 3600, n_times_between_points

            if between_points_idxs is None or between_points_idxs < idx_in_traj - 1:
                between_points_idxs = idx_in_traj - 1
                n_times_between_points = 1
            else:
                n_times_between_points += 1

            cur_point_in_traj = point_at_time(a, b, n_times_between_points * time_offset)
            # print 'Moving from %s to %s' % (a[0:2], b[0:2]), datetime.timedelta(seconds=time_clock), is_moving
            # print 'Moving to %s ' % (cur_point_in_traj[0:2]), datetime.timedelta(seconds=time_clock), is_moving

        if is_moving and time_clock >= arrival_time:
            is_moving = False
            traj_idx += 1
            idx_in_traj = 0
            n_times_between_points = 0
            between_points_idxs = None

            if traj_idx < len(real_model):
                next_tid = sorted(real_model, key=history_order_dict.get)[traj_idx]
                cur_loc = real_model[next_tid].start_point()

                leaving_time = (cur_loc[2] / 1000) % 86400 + tzoffset * 3600
                arrival_time = (real_model[next_tid].end_point()[2] / 1000) % 86400 + tzoffset * 3600

            # print 'Arriving in %s at' % cur_loc[:2], datetime.timedelta(seconds=time_clock), is_moving

        if time_clock < time_stop:
            if is_moving:
                lon = cur_point_in_traj[0]
                lat = cur_point_in_traj[1]
                agenda[time_clock] = [time_clock, lat, lon, 'move']
            else:
                lon = cur_loc[0]
                lat = cur_loc[1]
                agenda[time_clock] = [time_clock, lat, lon, 'stop']

    return agenda


def main():

    params = rome_params

    con = get_connection()
    cur = con.cursor()

    input_table = params['input_table']
    min_lat = params['min_lat']
    min_lon = params['min_lon']
    tzoffset = params['tzoffset']

    uid = 723576  # 146099 # rome
    # uid = 659447 # london
    week_limit = 42
    traintest_date = datetime.datetime.strptime('2015-05-03', '%Y-%m-%d')

    imh = load_individual_mobility_history(cur, uid, input_table)

    history_order_dict = get_ordered_history(imh)

    # train, test = train_test_partition_weeklimit(imh, history_order_dict, week_limit)
    # train, test, test_days = train_test_partition_date(imh, history_order_dict, date=traintest_date)
    train, test, test_days = train_test_partition_percentage(imh, history_order_dict, perc=0.7)
    day = 295


    # uid = 146099  # rome
    # #uid = 659447  # london
    # week_limit = 42
    #
    # imh = load_individual_mobility_history(cur, uid, input_table)
    #
    # history_order_dict = get_ordered_history(imh)
    #
    # train, test = train_test_partition_weeklimit(imh, history_order_dict, week_limit)
    #
    # day = 245
    # real_model1 = build_real_model(train, history_order_dict, day)
    # # ground_truth_model2 = build_ground_truth(test, history_order_dict)
    # #
    real_model1 = build_real_model(test, history_order_dict, day)

    print real_model1.keys()

    real_agenda = generate_real_agenda(real_model1, history_order_dict, tzoffset,
                                       time_start=0, time_offset=300, time_stop=86400)

    #
    # print '------'
    #
    # for event_time in sorted(real_agenda):
    #     event = real_agenda[event_time]
    #     print datetime.timedelta(seconds=event_time), event[1], event[2], event[3]
    #
    # print '-------'
    #
    # moving = False
    # for idx, event_time in enumerate(sorted(real_agenda)):
    #     event = real_agenda[event_time]
    #     if not moving and event[3] == 'move':
    #         print '%s,%s' % (real_agenda[sorted(real_agenda)[idx-1]][1], real_agenda[sorted(real_agenda)[idx-1]][2])
    #         moving = True
    #     if moving and event[3] == 'move':
    #         print '%s,%s' % (event[1], event[2])
    #     if moving and event[3] == 'stop':
    #         print '%s,%s' % (event[1], event[2])
    #         moving = False

if __name__ == "__main__":
    main()
