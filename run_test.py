from os import system

if __name__ == '__main__':
    to_run = [
        'kalman_filters_test.py', 'choldate_test.py',
        'qr_decomposition_test.py', 'sigma_points_test.py',
        'transition_functions_test.py', 'model_spec_processor_test.py',
        'data_processor_test.py', 'parse_params_test.py', 'chs_model_test.py']

    for file in to_run:
        system('python {}'.format(file))
