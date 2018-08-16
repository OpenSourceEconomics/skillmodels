from os import system

if __name__ == '__main__':
    normal_tests = [
        'fast_routines/transform_sigma_points_test',
        'estimation/skill_model_test', 'estimation/parse_params_test',
        'estimation/wa_functions_test', 'fast_routines/kalman_filters_test',
        'fast_routines/choldate_test', 'fast_routines/qr_decomposition_test',
        'fast_routines/sigma_points_test',
        'model_functions/transition_functions_test',
        'pre_processing/model_spec_processor_test',
        'pre_processing/data_processor_test']

    long_running_tests = [
        'estimation/wa_test_with_no_squares_translog_model']

    to_run = normal_tests + long_running_tests
    for file in to_run:

        system('python skillmodels/tests/{}.py'.format(file))
