from os import system

if __name__ == '__main__':
    normal_tests = [
        'wa_functions_test', 'kalman_filters_test', 'choldate_test',
        'qr_decomposition_test', 'sigma_points_test',
        'transition_functions_test', 'model_spec_processor_test',
        'data_processor_test', 'parse_params_test', 'skill_model_test']

    long_running_tests = [
        'wa_test_with_no_squares_translog_model']

    to_run = normal_tests + long_running_tests
    # to_run = long_running_tests
    # to_run = ['wa_functions_test', 'wa_test_with_no_squares_translog_model']

    for file in to_run:
        system('python {}.py'.format(file))
