from os import system

if __name__ == '__main__':
    normal_tests = [
        'skill_model_results_test',
        'skill_model_test', 'parse_params_test',
        'wa_functions_test', 'kalman_filters_test', 'choldate_test',
        'qr_decomposition_test', 'sigma_points_test',
        'transition_functions_test', 'model_spec_processor_test',
        'data_processor_test']

    long_running_tests = [
        'wa_test_with_no_squares_translog_model']

    to_run = normal_tests + long_running_tests
    for file in to_run:
        system('python {}.py'.format(file))
