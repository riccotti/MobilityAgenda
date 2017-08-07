import numpy as np
from util.mobility_distance_functions import spherical_distance


def evaluate_agenda(real_agenda, my_agenda, spt_tol, tmp_tol, time_start=0, time_offset=300, time_stop=86400):

    # dato il real_point devo trovare sul percorso reale il punto X
    # piu vicino a real_point tale che dist_temp(X, real_point) < tmp_tol
    # poi verificare che dist_spat(X, pred_point) < spt_tol

    spt_err = list()
    spt_err_stop = list()
    spt_err_move = list()

    n_predictions = len(real_agenda)
    n_predictions_stop = 0
    n_predictions_move = 0

    n_correct_predictions = 0
    n_correct_predictions_stop = 0
    n_correct_predictions_move = 0

    re_stop = 0
    re_move = 0
    my_stop = 0
    my_move = 0

    ss = 0
    sm = 0
    ms = 0
    mm = 0

    for event_time in sorted(real_agenda):
        real_point = real_agenda[event_time][1:3]
        my_point = my_agenda[event_time][1:3]

        real_status = real_agenda[event_time][3]
        my_status = my_agenda[event_time][3]

        if my_status is 'stop':
            n_predictions_stop += 1

        if my_status is 'move':
            n_predictions_move += 1

        # if real_status is 'stop':
        #     n_predictions_stop += 1
        #
        # if real_status is 'move':
        #     n_predictions_move += 1

        if tmp_tol > 0:
            et_start = max(time_start, event_time - tmp_tol)
            et_end = min(time_stop, event_time + tmp_tol)

            min_dist = float('inf')
            for event_time2 in range(et_start, et_end, time_offset):
                my_point = my_agenda[event_time2][1:3]
                dist = spherical_distance(real_point, my_point)
                if dist < min_dist:
                    min_dist = dist

            if min_dist < spt_tol:
                n_correct_predictions += 1
                spt_err.append(min_dist)
                if my_status is 'stop':  # and my_status is 'stop':
                    n_correct_predictions_stop += 1
                if my_status is 'move':  # and my_status is 'move':
                    n_correct_predictions_move += 1
                # if real_status is 'stop': #and my_status is 'stop':
                #     n_correct_predictions_stop += 1
                # if real_status is 'move': #and my_status is 'move':
                #     n_correct_predictions_move += 1

            dist = min_dist if min_dist <= spt_tol else spherical_distance(real_point, my_point)
            spt_err.append(dist)
            if my_status is 'stop':  # and my_status is 'stop':
                spt_err_stop.append(dist)
            if my_status is 'move':  # and my_status is 'move':
                spt_err_move.append(dist)
            # if real_status is 'stop': #and my_status is 'stop':
            #     spt_err_stop.append(dist)
            # if real_status is 'move': #and my_status is 'move':
            #     spt_err_move.append(dist)

        else:
            my_point = my_agenda[event_time][1:3]
            dist = spherical_distance(real_point, my_point)
            spt_err.append(dist)
            if my_status is 'stop':
                spt_err_stop.append(dist)
            if my_status is 'move':
                spt_err_move.append(dist)
            # if real_status is 'stop':
            #     spt_err_stop.append(dist)
            # if real_status is 'move':
            #     spt_err_move.append(dist)

            if dist <= spt_tol:
                n_correct_predictions += 1
                if real_status is 'stop':
                    n_correct_predictions_stop += 1
                if real_status is 'move':
                    n_correct_predictions_move += 1

        if real_status is 'stop' and my_status is 'stop':
            re_stop += 1
            my_stop += 1
            ss += 1
        elif real_status is 'stop' and my_status is 'move':
            re_stop += 1
            my_move += 1
            ms += 1
        elif real_status is 'move' and my_status is 'stop':
            re_move += 1
            my_stop += 1
            sm += 1
        elif real_status is 'move' and my_status is 'move':
            re_move += 1
            my_move += 1
            mm += 1

    acc_rate = 1.0 * n_correct_predictions / n_predictions
    acc_rate_stop = 1.0 * n_correct_predictions_stop / n_predictions_stop
    acc_rate_move = 0.0 if n_predictions_move == 0 else 1.0 * n_correct_predictions_move / n_predictions_move
    acc_rate_f1 = 0.0 if (acc_rate_stop + acc_rate_move) == 0 \
        else 2.0 * acc_rate_stop * acc_rate_move / (acc_rate_stop + acc_rate_move)

    stop_pre = 0.0 if (ss + sm) == 0 else 1.0 * ss / (ss + sm)
    stop_rec = 0.0 if re_stop == 0 else 1.0 * ss / re_stop
    stop_f1 = 0.0 if (stop_pre + stop_rec) == 0 else 2.0 * stop_pre * stop_rec / (stop_pre + stop_rec)

    move_pre = 0.0 if (mm + ms) == 0 else 1.0 * mm / (mm + ms)
    move_rec = 0.0 if re_move == 0 else 1.0 * mm / re_move
    move_f1 = 0.0 if (move_pre + move_rec) == 0 else 2.0 * move_pre * move_rec / (move_pre + move_rec)

    avg_f1 = (stop_f1 + move_f1) / 2.0

    res = {
        'acc_rate': acc_rate,
        'acc_rate_f1': acc_rate_f1,
        'acc_rate_stop': acc_rate_stop,
        'acc_rate_move': acc_rate_move,
        'avg_spt_err': np.mean(spt_err),
        'std_spt_err': np.std(spt_err),
        'med_spt_err': np.median(spt_err),
        'avg_spt_err_stop': np.mean(spt_err_stop),
        'std_spt_err_stop': np.std(spt_err_stop),
        'med_spt_err_stop': np.median(spt_err_stop),
        'avg_spt_err_move': np.mean(spt_err_move),
        'std_spt_err_move': np.std(spt_err_move),
        'med_spt_err_move': np.median(spt_err_move),
        'stop_pre': stop_pre,
        'stop_rec': stop_rec,
        'stop_f1': stop_f1,
        'move_pre': move_pre,
        'move_rec': move_rec,
        'move_f1': move_f1,
        'avg_f1': avg_f1
    }

    return res


def evaluate_agenda2(real_agenda, my_agenda, spt_tol, tmp_tol, time_start=0, time_offset=300, time_stop=86400):

        # dato il real_point devo trovare sul percorso reale il punto X
        # piu vicino a real_point tale che dist_temp(X, real_point) < tmp_tol
        # poi verificare che dist_spat(X, pred_point) < spt_tol

        spt_err = list()
        spt_err_stop = list()
        spt_err_move = list()
        spt_err_change = list()

        n_predictions = 0
        n_predictions_stop = 0
        n_predictions_move = 0
        n_change_predictions = 0

        n_correct_predictions = 0
        n_correct_predictions_stop = 0
        n_correct_predictions_move = 0
        n_correct_change_predictions = 0

        previous_status = None
        previous_status2 = None

        for event_time in sorted(real_agenda):
            real_point = real_agenda[event_time][1:3]
            my_point = my_agenda[event_time][1:3]

            real_status = real_agenda[event_time][3]
            my_status = my_agenda[event_time][3]

            n_predictions += 1

            if my_status is 'stop':
                n_predictions_stop += 1

            if my_status is 'move':
                n_predictions_move += 1

            is_changing_position = False
            if previous_status is not None:
                if previous_status is 'move' or real_status is 'move' \
                        or previous_status2 is 'move' or my_status is 'move':
                    is_changing_position = True
                    n_change_predictions += 1

            previous_status = real_status
            previous_status2 = my_status

            if tmp_tol > 0:
                et_start = max(time_start, event_time - tmp_tol)
                et_end = min(time_stop, event_time + tmp_tol)

                min_dist = float('inf')
                for event_time2 in range(et_start, et_end, time_offset):
                    my_point2 = my_agenda[event_time2][1:3]
                    dist = spherical_distance(real_point, my_point2)
                    if dist < min_dist:
                        min_dist = dist

                if min_dist < spt_tol:
                    n_correct_predictions += 1
                    spt_err.append(min_dist)
                    if my_status is 'stop':
                        n_correct_predictions_stop += 1
                    if my_status is 'move':
                        n_correct_predictions_move += 1
                    if is_changing_position:
                        n_correct_change_predictions += 1

                if min_dist <= spt_tol:
                    dist = min_dist
                else:
                    dist = spherical_distance(real_point, my_point)
                spt_err.append(dist)

                if my_status is 'stop':
                    spt_err_stop.append(dist)
                if my_status is 'move':
                    spt_err_move.append(dist)
                if is_changing_position:
                    # print event_time/3600, (event_time%3600)/60, real_point, my_point, dist, my_status, real_status
                    spt_err_change.append(dist)

            else:

                dist = spherical_distance(real_point, my_point)
                spt_err.append(dist)

                if my_status is 'stop':
                    spt_err_stop.append(dist)
                if my_status is 'move':
                    spt_err_move.append(dist)
                if is_changing_position:
                    spt_err_change.append(dist)

                if dist <= spt_tol:
                    n_correct_predictions += 1
                    if my_status is 'stop':
                        n_correct_predictions_stop += 1
                    if my_status is 'move':
                        n_correct_predictions_move += 1
                    if is_changing_position:
                        spt_err_change.append(dist)

        acc_rate = 1.0 * n_correct_predictions / max(n_predictions, 1)
        acc_rate_stop = 1.0 * n_correct_predictions_stop / max(n_predictions_stop, 1)
        acc_rate_move = 1.0 * n_correct_predictions_move / max(n_predictions_move, 1)
        acc_rate_f1 = 0.0 if (acc_rate_stop + acc_rate_move) == 0 \
            else 2.0 * acc_rate_stop * acc_rate_move / (acc_rate_stop + acc_rate_move)
        acc_rate_change = 1.0 * n_correct_change_predictions / max(n_change_predictions, 1)

        res = {
            'acc_rate': acc_rate,
            'acc_rate_f1': acc_rate_f1,
            'acc_rate_stop': acc_rate_stop,
            'acc_rate_move': acc_rate_move,
            'acc_rate_change': acc_rate_change,
            'spt_err': np.median(spt_err),
            'spt_err_stop': np.median(spt_err_stop),
            'spt_err_move': np.median(spt_err_move),
            'spt_err_change': np.median(spt_err_change)
        }

        return res


def evaluate_agenda3(real_agenda, my_agenda, spt_tol, tmp_tol, time_start=0, time_offset=300, time_stop=86400):

    # dato il real_point devo trovare sul percorso reale il punto X
    # piu vicino a real_point tale che dist_temp(X, real_point) < tmp_tol
    # poi verificare che dist_spat(X, pred_point) < spt_tol

    spt_err = list()
    spt_err_stop = list()
    spt_err_move = list()

    n_predictions = 0
    n_predictions_stop = 0
    n_predictions_move = 0

    n_correct_predictions = 0
    n_correct_predictions_stop = 0
    n_correct_predictions_move = 0

    sorted_real_agenda = sorted(real_agenda)
    next_is_arrival = False

    for event_time_idx in range(0, len(sorted_real_agenda)-1):
        event_time_idx1 = event_time_idx + 1

        event_time = sorted_real_agenda[event_time_idx]
        event_time1 = sorted_real_agenda[event_time_idx1]

        real_point = real_agenda[event_time][1:3]
        my_point = my_agenda[event_time][1:3]

        real_status = real_agenda[event_time][3]
        my_status = my_agenda[event_time][3]

        real_status1 = real_agenda[event_time1][3]
        my_status1 = my_agenda[event_time1][3]

        # if (real_status is 'stop' and real_status1 is 'move') or \
        #         (real_status is 'move') or next_is_arrival:

        if (real_status is 'stop' and real_status1 is 'move') or (my_status is 'stop' and my_status1 is 'move') or \
                (real_status is 'move') or (my_status is 'move') or next_is_arrival:

            # if (real_status is 'move' and real_status1 is 'stop'):
            if (real_status is 'move' and real_status1 is 'stop') or (my_status is 'move' and my_status1 is 'stop'):
                next_is_arrival = True
            else:
                next_is_arrival = False

            n_predictions += 1

            if my_status is 'stop':
                n_predictions_stop += 1

            if my_status is 'move':
                n_predictions_move += 1

            if tmp_tol > 0:
                et_start = max(time_start, event_time - tmp_tol)
                et_end = min(time_stop, event_time + tmp_tol)

                min_dist = float('inf')
                for event_time2 in range(et_start, et_end, time_offset):
                    my_point2 = my_agenda[event_time2][1:3]
                    dist = spherical_distance(real_point, my_point2)
                    if dist < min_dist:
                        min_dist = dist

                if min_dist < spt_tol:
                    n_correct_predictions += 1
                    spt_err.append(min_dist)
                    if my_status is 'stop':
                        n_correct_predictions_stop += 1
                    if my_status is 'move':
                        n_correct_predictions_move += 1

                if min_dist <= spt_tol:
                    dist = min_dist
                else:
                    dist = spherical_distance(real_point, my_point)
                spt_err.append(dist)

                if my_status is 'stop':
                    spt_err_stop.append(dist)
                if my_status is 'move':
                    spt_err_move.append(dist)

            else:

                dist = spherical_distance(real_point, my_point)
                spt_err.append(dist)

                if my_status is 'stop':
                    spt_err_stop.append(dist)
                if my_status is 'move':
                    spt_err_move.append(dist)

                if dist <= spt_tol:
                    n_correct_predictions += 1
                    if my_status is 'stop':
                        n_correct_predictions_stop += 1
                    if my_status is 'move':
                        n_correct_predictions_move += 1

    acc_rate = 1.0 * n_correct_predictions / max(n_predictions, 1)
    acc_rate_stop = 1.0 * n_correct_predictions_stop / max(n_predictions_stop, 1)
    acc_rate_move = 1.0 * n_correct_predictions_move / max(n_predictions_move, 1)
    acc_rate_f1 = 2.0 * acc_rate_stop * acc_rate_move / max(acc_rate_stop + acc_rate_move, 1)

    res = {
        'acc_rate': acc_rate,
        'acc_rate_f1': acc_rate_f1,
        'acc_rate_stop': acc_rate_stop,
        'acc_rate_move': acc_rate_move,
        'spt_err': np.median(spt_err_stop + spt_err_move),
        'spt_err_stop': np.median(spt_err_stop),
        'spt_err_move': np.median(spt_err_move),
        'n_predictions': n_predictions,
        'n_predictions_stop': n_predictions_stop,
        'n_predictions_move': n_predictions_move,
        'n_correct_predictions': n_correct_predictions,
        'n_correct_predictions_stop': n_correct_predictions_stop,
        'n_correct_predictions_move': n_correct_predictions_move
    }

    return res

    # print acc_rate
    # print np.mean(spt_err), np.std(spt_err), np.median(spt_err)
    # print np.mean(spt_err_stop), np.std(spt_err_stop), np.median(spt_err_stop)
    # print np.mean(spt_err_move), np.std(spt_err_move), np.median(spt_err_move)
    #
    # print stop_pre, stop_rec, stop_f1
    # print move_pre, move_rec, move_f1
    # print avg_f1

    # print ''
    # print 're_stop', re_stop
    # print 're_move', re_move
    # print 'my_stop', my_stop
    # print 'my_move', my_move
    # print 'ss', ss
    # print 'sm', sm
    # print 'ms', ms
    # print 'mm', mm




