# Universal-science-and-technology-
Focus on data analysis, algorithm research, robot research and other technology development。

def option(series, start):
    data = np.where(series < start, series, np.nan)
    new_series = pd.Series(data)
    new_series[len(data) +1] = 1 if not new_series.isnull().any() else 0

    return new_series
