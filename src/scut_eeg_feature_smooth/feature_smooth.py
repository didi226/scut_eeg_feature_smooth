import numpy as np
from pykalman import UnscentedKalmanFilter

def moving_average_filter(data, window_size):
    filtered_data = []
    window = [0] * window_size
    for i in range(len(data)):
        window.pop(0)
        window.append(data[i])
        filtered_data.append(sum(window) / window_size)
    filtered_data = np.array(filtered_data)
    return filtered_data


def lsd_KalmanFilter(data, window_size):
    from pykalman import KalmanFilter
    window_num = data.shape[0] // window_size
    smoothed_feature = []
    for i_window in range(window_num + 1):
        begin_idx = window_size * i_window
        end_idx = window_size * (i_window + 1)
        if begin_idx >= data.shape[0]:
            continue
        if end_idx > data.shape[0]:
            end_idx = data.shape[0]
        data_window = data[begin_idx:end_idx]
        transition_covariance = np.diag([0.1, 0.1])
        transition_covariance = 0.1
        observation_covariance = 0.001
        initial_state_mean = np.mean(data_window)
        initial_state_covariance = 1
        kf = KalmanFilter(transition_covariance=transition_covariance,
                          observation_covariance=observation_covariance,
                          initial_state_mean=initial_state_mean,
                          initial_state_covariance=initial_state_covariance)
        # Estimate the parameters using the EM algorithm
        kf = kf.em(data_window)
        estimated_A = kf.transition_matrices
        estimated_C = kf.observation_matrices
        estimated_Q = kf.transition_covariance
        estimated_R = kf.observation_covariance

        smoothed_state_means, smoothed_state_covs = kf.smooth(data_window)
        smoothed_feature.extend(smoothed_state_means.flatten())
    smoothed_feature = np.array(smoothed_feature)
    return smoothed_feature



def lsd_UnscentedKalmanFilter(data, window_size, observation_functions_type=None):

    window_num = data.shape[0] // window_size
    smoothed_feature = []
    if observation_functions_type == "sigmoid":
        def measurement_function(x, w):
            return np.arctanh(x / (5 * 10 ** 7)) * 10 ** 7 + w

        def measurement_function_oly_x(x):
            return np.arctanh(x / (5 * 10 ** 7)) * 10 ** 7
    else:
        measurement_function = None
    for i_window in range(window_num + 1):
        begin_idx = window_size * i_window
        end_idx = window_size * (i_window + 1)
        if begin_idx >= data.shape[0]:
            continue
        if end_idx > data.shape[0]:
            end_idx = data.shape[0]
        data_window = data[begin_idx:end_idx]

        transition_covariance = 0.1
        observation_covariance = 0.001
        if observation_functions_type == "sigmoid":
            initial_state = [measurement_function_oly_x(x) for x in data_window]
            initial_state_mean = np.mean(initial_state)
        else:
            initial_state_mean = np.mean(data_window)
        initial_state_covariance = 1
        kf = UnscentedKalmanFilter(

            observation_functions=measurement_function,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance
        )

        smoothed_state_means, smoothed_state_covs = kf.smooth(data_window)
        smoothed_feature.extend(smoothed_state_means.flatten())
    smoothed_feature = np.array(smoothed_feature)
    return smoothed_feature


def feature_smooth(data, smooth_type="lds", window_size=10):
    """
    Args:
        refernce
        [1]Duan R N, Zhu J Y, Lu B L. Differential entropy feature for EEG-based emotion classification[C]//2013
        6th International IEEE/EMBS Conference on Neural Engineering (NER). IEEE, 2013: 81-84.
        [2]Shi L C, Lu B L. Off-line and on-line vigilance estimation based on linear dynamical system and manifold
        learning[C]//2010 Annual International Conference of the IEEE Engineering in Medicine and Biology. IEEE, 2010: 6587-6590.
        [3]Zheng W L, Zhu J Y, Lu B L. Identifying stable patterns over time for emotion recognition from
        EEG[J]. IEEE Transactions on Affective Computing, 2017, 10(3): 417-429.

        data:                        narray      shape (n_eopoch,n_channel,n_feature)
        smooth_type:                 str         "mv_av_filter"   move average filter
                                                 "lds"             linear dynamic system (LDS) approach
                                                 "NDS-UKF"       non-linear dynamic system UnscentedKalmanFilter
        window_size:                 int

    Returns:

    """
    n_eopoch, n_channel, n_feature = data.shape
    print(n_eopoch, n_channel, n_feature)
    smoothed_data = np.zeros((n_eopoch, n_channel, n_feature))
    for i_feature in range(n_feature):
        for i_channel in range(n_channel):
            print(data[:, i_channel, i_feature].shape)
            if smooth_type == "mv_av_filter":
                smoothed_data[:, i_channel, i_feature] = moving_average_filter(data[:, i_channel, i_feature],
                                                                                    window_size)
            elif smooth_type == "lds":
                smoothed_data[:, i_channel, i_feature] = lsd_KalmanFilter(data[:, i_channel, i_feature],
                                                                               window_size)
            # if smooth_type == "UnscentedKalmanFilter":
            #     smoothed_data[:, i_channel, i_feature] = lsd_UnscentedKalmanFilter(data[:, i_channel, i_feature],
            #                                                                             window_size)
            elif smooth_type == "NDS-UKF":
                smoothed_data[:, i_channel, i_feature] = lsd_UnscentedKalmanFilter(data[:, i_channel, i_feature],
                                                                                        window_size, "sigmoid")
            else:
                raise ValueError(f"smooth_type of {smooth_type}  does not exist")
    return smoothed_data