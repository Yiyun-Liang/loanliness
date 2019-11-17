# x = pd.DataFrame(data=x[:, 1:], index=x[:, 0])
    # print(x[0])
    # print(np.sum(np.isnan(x[0])))
    # raise
    # x = x.replace([np.inf, -np.inf], np.nan)
    # print(type(x))
    # print(x.shape)
    # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    # imp = imp.fit(x)
    # # print(x.shape)
    # x = imp.transform(x)
    # print(x[0])
    # x.astype(np.float64)
    # x[x >= 10 ** 10] = 10 ** 10
    # x[x <= -1 * (10 ** 10)] = -1 * (10 ** 10)
    # x.astype(np.float64)
    # print(np.any(np.isnan(x)))
    # print(np.all(np.isfinite(x)))
    # raise
    # print(x.shape)
    # raise
    # print(type(x))
    # x = x.to_numpy()
    # x = x[:, 1:]
    # print(x)
    # print(x[1, :])
    # raise
    # print(x[:, 1])
    # print(x[0, :])
    # raise
    # rank_x = np.unique(x)
    # x = x.replace([np.inf], np.nan)
    # np.where(x==np.inf, np.nan, x)
    # np.where(x==-1 * np.inf, np.nan, x)
    # x[x == np.inf] = np.nan
    # x[x == -1 * np.inf] = np.nan
    # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    # imp = imp.fit(x)
    # x = imp.transform(x)
    # x[x >= 10 ** 8] = 10 ** 8
    # x[x <= -1 * 10 ** 8] = -1 * 10 ** 8
    # np.where(x==np.inf, np.mean(x), x)
    # np.where(x==-np.inf, np.mean(x), x)
    # imp = SimpleImputer(missing_values=np.inf, strategy='mean')
    # imp = imp.fit(x)
    # x = imp.transform(x)
    # imp = SimpleImputer(missing_values=-1 * np.inf, strategy='mean')
    # imp = imp.fit(x)
    # x = imp.transform(x)


def _data_preprocessing(entries, data, test = False):
    print(len(entries))
    print(data.shape)
    print()
    raise
    value_dict = dict()               # Store the possible values of the col
    processed_data = np.copy(data)
    if not test:
        start_entry = 2
    else:
        start_entry = 1
    remove_row_ls = []
    remove_col_ls = []
    for i in range(start_entry, data.shape[1]):
        print(i)
        # Assign values to string elements
        '''
        col = data[:, i]
        possible_values = sorted(get_possible_values(col))
        if 'nan' in possible_values:
            possible_values.remove('nan')
        if len(possible_values) < 100:
            value_dict[entries[i]] = possible_values
            for j in range(data.shape[0]):
                if data[j, i] != 'nan':
                    value = possible_values.index(data[j, i])
                    processed_data[j, i] = value
        '''
    print('Preprocessing: value assignment has been finished.')
    for i in range(start_entry, data.shape[1]):
        # Add the column with too many empty entries to remove_col_ls
        col = data[:, i]
        # if count_empty_percentage(col) > 0.1:
        print(count_nan_percentage(col))
        print(Counter(col)['nan'])
        print(col)
        if count_nan_percentage(col) > 0.1:
            remove_col_ls.append(i)
        # Add the row with empty entries to remove_row_ls
        #elif count_empty_percentage(col) != 0:
        elif count_nan_percentage(col) != 0:
            print('nan is 0 percent')
            for j in range(data.shape[0]):
                if data[j, i] == 'nan':
                    remove_row_ls.append(j)
    print('Preprocessing: removal row and col numbers has been stored.')
    # Remove the data points and features which do not satisfy requiremetns
    remove_col_set = set(remove_col_ls)
    remove_row_set = set(remove_row_ls)
    print(len(remove_row_set))
    revised_entries = np.copy(entries)
    processed_data = np.delete(processed_data, list(remove_col_set), 1)
    print('Preprocessing: cols have been removed.')
    processed_data = np.delete(processed_data, list(remove_row_set), 0)
    print('Preprocessing: rows have been removed.')
    np.delete(revised_entries, list(remove_col_set))
    print('Preprocessing: preprocessing has been finished.')
    return processed_data.astype(np.float), revised_entries, value_dict

