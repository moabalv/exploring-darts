from darts import FFT
from darts.metrics import mae

def get_best_frequency(training, validation, matches, trend, trend_degree, frequency_limit):
    """
    Return the frequency number that has the lowest mean absolute error in the Fast Fourier Transform Model for a given time series
    
    Args:
        training (TimeSeries)
        validation (TimeSeries)
        matches (Set): time index names
        trend (String)
        trend_degree (Int): 0 if the series has no trend
        frequency_limit (Int): max frequency to be tested 
    """
    model = FFT(required_matches = matches,
                nr_freqs_to_keep = 1,
                trend = trend,
                trend_poly_degree = trend_degree)
    model.fit(training)
    predicted = model.predict(len(validation))
    mae_value = mae(validation, predicted)
    best_i = 1

    for i in range(2, frequency_limit):
        model = FFT(required_matches = matches,
                nr_freqs_to_keep = i,
                trend = trend,
                trend_poly_degree = trend_degree)
        model.fit(training)
        predicted = model.predict(len(validation))
        current_mae = mae(validation, predicted)
        if current_mae < mae_value:
            mae_value = current_mae
            best_i = i
    
    return predicted, best_i, mae_value
