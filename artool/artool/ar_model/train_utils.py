from sklearn.model_selection import train_test_split


def data_split_by_time(df, time_var, split_time, val_ratio=0.2, random_state=42):
    df_train = df[df[time_var] < split_time]
    df_test = df[df[time_var] >= split_time]
    if val_ratio <= 0:
        return df_train, None, df_test
    # separate validation
    df_train, df_val = train_test_split(
        df_train, test_size=val_ratio, random_state=random_state
    )
    return df_train, df_val, df_test
