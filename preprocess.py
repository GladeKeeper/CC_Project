def initial_process(_input_dir, _output_dir="output"):
    """
    :param _input_dir: input directory name
                      AWS S3 directory name, where the input files are stored
    :param _output_dir: output directory name
                      AWS S3 directory name, where the output files are saved
    :return:
            the processed data, and demand data
    """
    import os.path
    import pandas as pd
    from uszipcode import SearchEngine

    # to obtain the zip-codes for latitude, longitude values
    engine = SearchEngine()

    if not os.path.exists(_output_dir):
        os.makedirs(_output_dir)

    def get_zip(entry):
        return engine.by_coordinates(entry['Lat'], entry['Lon'], radius=10, returns=1)[0].zipcode

    def get_zip_codes(df):
        zip_codes = []
        zip_history = {}
        for i, entry in df.iterrows():
            if i % 1000 == 0:
                print(i)
            if (entry['Lat'], entry['Lon']) in zip_history:
                zip_codes.append(zip_history[entry['Lat'], entry['Lon']])
            else:
                zip_code = get_zip(entry)
                zip_history[entry['Lat'], entry['Lon']] = zip_code
                zip_codes.append(zip_code)
        return zip_codes

    # load all the data
    months = ["sep"]

    file_format = "uber-raw-data-{}14.csv"
    for month in months:
        file_name = _input_dir + "/" + file_format.format(month)
        _data = pd.read_csv(file_name)
        # obtaining the zip-codes
        _data['zip'] = get_zip_codes(_data)
        # process date and time
        _data['Date/Time'] = pd.to_datetime(_data['Date/Time'], format='%m/%d/%Y %H:%M:%S')
        _data['weekday'] = _data['Date/Time'].dt.dayofweek
        _data['hour'] = _data['Date/Time'].dt.hour

        output_file_name = _input_dir + "/" + file_format.format(month)
        output_file_name = output_file_name.replace("raw", "processed")
        _data.to_csv(output_file_name, index=False)


from datetime import datetime
start = datetime.now()
initial_process("data", "output")
end = datetime.now()
print(end)
print(start)
print(f"initial processing time {(end - start).total_seconds()}")
