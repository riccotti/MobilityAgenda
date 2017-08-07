from method.home_agenda import *
from method.homework_agenda import *
from method.random_agenda import *
from method.grid_agenda import *
from method.my_agenda import *
from method.real_agenda import *

from evaluation.evaluation_metrics import *


__author__ = 'Riccardo Guidotti'


def main():

    params = rome_params

    con = get_connection()
    cur = con.cursor()

    input_table = params['input_table']
    min_lat = params['min_lat']
    min_lon = params['min_lon']
    tzoffset = params['tzoffset']

    uid = 1
    week_limit = 42
    traintest_date = datetime.datetime.strptime('2015-05-03', '%Y-%m-%d')

    imh = load_individual_mobility_history(cur, uid, input_table)

    history_order_dict = get_ordered_history(imh)

    # train, test = train_test_partition_weeklimit(imh, history_order_dict, week_limit)
    # train, test, test_days = train_test_partition_percentage(imh, history_order_dict, perc=0.7)
    train, test, test_days = train_test_partition_date(imh, history_order_dict, date=traintest_date)

    grid_length = 250
    time_lenght = 1800

    print '\nReal'

    day = test_days[np.random.randint(0, min(len(test_days), 10))]
    day = 136
    print 'Predicted day', day

    real_model1 = build_real_model(test, history_order_dict, day)
    real_agenda = generate_real_agenda(real_model1, history_order_dict, tzoffset,
                                       time_start=0, time_offset=300, time_stop=86400)

    print 'Grid'
    st_grid_model = build_st_grid(train, history_order_dict, grid_length, time_lenght,
                                  min_lon, min_lat, tzoffset)
    st_grid_agenda, _ = generate_st_grid_agenda(st_grid_model, grid_length, time_lenght, min_lon, min_lat, tzoffset,
                                             time_start=0, time_offset=300, time_stop=86400)

    print '\nHome'

    home_model = build_home(train, history_order_dict, grid_length, min_lon, min_lat)

    home_agenda = generate_home_agenda(home_model, grid_length, min_lon, min_lat,
                                       time_start=0, time_offset=300, time_stop=86400)

    print '\nHomeWork'

    homework_model = build_homework(train, history_order_dict, grid_length, time_lenght,
                                    min_lon, min_lat, tzoffset)

    homework_agenda = generate_homework_agenda(homework_model, grid_length, time_lenght, min_lon, min_lat,
                                               time_start=0, time_offset=300, time_stop=86400)

    print '\nMyAgenda'

    print 'start', datetime.datetime.now()
    myagenda_model = build_myagenda(train, tzoffset, reg_loc=True)
    print 'end', datetime.datetime.now()

    my_agenda, _ = generate_myagenda(myagenda_model, random_choice=False,
                                     time_start=0, time_offset=300, time_stop=86400)

    print ''

    spt_tol = 250  # meters
    tmp_tol = 5 * 60  # seconds

    print 'Grid Eval'
    eval1 = evaluate_agenda2(real_agenda, st_grid_agenda, spt_tol, tmp_tol,
                             time_start=0, time_offset=5 * 60, time_stop=86400)

    print '\nHome Eval'
    eval3 = evaluate_agenda2(real_agenda, home_agenda, spt_tol, tmp_tol,
                             time_start=0, time_offset=5 * 60, time_stop=86400)

    print '\nHomeWork Eval'
    eval4 = evaluate_agenda2(real_agenda, homework_agenda, spt_tol, tmp_tol,
                             time_start=0, time_offset=5 * 60, time_stop=86400)

    print '\nMyAgenda Eval'
    eval5 = evaluate_agenda2(real_agenda, my_agenda, spt_tol, tmp_tol,
                             time_start=0, time_offset=5 * 60, time_stop=86400)

    for ek in sorted(eval5):
        print ek, eval1[ek],  eval3[ek], eval4[ek], eval5[ek]


if __name__ == "__main__":
    main()

