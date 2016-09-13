from os import system

if __name__ == '__main__':
    to_run = [
        'wa_functions_test', 'kalman_filters_test', 'choldate_test',
        'qr_decomposition_test', 'sigma_points_test',
        'transition_functions_test', 'model_spec_processor_test',
        'data_processor_test', 'parse_params_test', 'skill_model_test',
        'wa_test_with_no_squares_translog_model']

    for file in to_run:
        system('python {}.py'.format(file))
