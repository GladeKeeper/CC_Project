import os
import warnings
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.models import model_from_json, load_model

warnings.filterwarnings('ignore')
NUM_SAMPLES = 5


def load_data(input_dir, output_dir="output", data_size=-1):
    """
    :param input_dir: input directory name
                      AWS S3 directory name, where the input files are stored
    :param output_dir: output directory name
                      AWS S3 directory name, where the output files are saved
    :param data_size: size of data
                      Data size, that needs to be tested, by default it takes value of
                      -1, which means consider all the data
    :return:
            the processed data, and demand data
    """
    import os.path
    import pandas as pd
    from pandas import DataFrame
    from uszipcode import SearchEngine

    # to obtain the zip-codes for latitude, longitude values
    engine = SearchEngine()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load all the data
    months = ["apr", "may", "jun", "jul", "aug", "sep"]
    file_format = "uber-raw-data-{}14.csv"
    df = DataFrame()
    for month in months:
        file_name = input_dir + "/" + file_format.format(month)
        df_sub = pd.read_csv(file_name)
        df = df.append(df_sub)

    # sample the data
    if data_size > 0:
        df = df.sample(n=data_size)

    # process date and time
    df['Date/Time'] = pd.to_datetime(df['Date/Time'], format='%m/%d/%Y %H:%M:%S')
    df['month'] = df['Date/Time'].dt.month
    df['weekday'] = df['Date/Time'].dt.dayofweek
    df['day'] = df['Date/Time'].dt.day
    df['hour'] = df['Date/Time'].dt.hour
    df['minute'] = df['Date/Time'].dt.minute
    df['lat_short'] = round(df['Lat'], 2)
    df['lon_short'] = round(df['Lon'], 2)

    # obtaining the zip-codes
    df['zip'] = df.apply(
        lambda row: engine.by_coordinates(row['Lat'], row['Lon'], radius=10)[0].zipcode, axis=1
    )

    # summarizing demand data
    demand = (df.groupby(['zip']).count()['Date/Time']).reset_index()
    demand.columns = ['Zip', 'Number of Trips']
    demand.to_csv(output_dir + "/demand.csv", index=False)

    demand_w = (df.groupby(['zip', 'weekday']).count()['Date/Time']).reset_index()
    demand_w.columns = ['Zip', 'Weekday', 'Number of Trips']
    demand_w.to_csv(output_dir + "/demand_dow.csv", index=False)

    demand_h = (df.groupby(['zip', 'hour']).count()['Date/Time']).reset_index()
    demand_h.columns = ['Zip', 'Hour', 'Number of Trips']
    demand_h.to_csv(output_dir + "/demand_h.csv", index=False)

    demand_wh = (df.groupby(['zip', 'weekday', 'hour']).count()['Date/Time']).reset_index()
    demand_wh.columns = ['Zip', 'Weekday', 'Hour', 'Number of Trips']
    demand_wh.to_csv(output_dir + "/demand_h_dow.csv", index=False)

    return df, demand, demand_w, demand_h, demand_wh


class DemandPredictorBase(object):
    """
        Base class for demand predictor
    """

    def __init__(self, _x, _y, train=True):
        self.x = _x
        self.y = _y
        self.model = self.build_model()
        if train:
            self.train()

    def build_model(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def predict(self, _x_test):
        """
        :param _x_test: test dataset
        :return: prediction for the given test dataset _x_test
        """
        return self.model.predict(_x_test)

    def predict_and_scale(self, _x_test, _y_scalar):
        """
        :param _x_test: test dataset
        :param _y_scalar: Scaler
        :return: prediction for the given test dataset _x_test, scaled to the scalar
        """
        return _y_scalar.inverse_transform(self.predict(_x_test))

    @staticmethod
    def get_mse(_y_test, _y_pred):
        """
        :param _y_test: actual test values
        :param _y_pred: predicted test values
        :return: return the mean square error
        """
        return mean_squared_error(_y_test, _y_pred)

    def get_mse_avg(self, _y_test, _y_pred):
        """
        :param _y_test: actual test values
        :param _y_pred: predicted test values
        :return: return the ratio between MSE and average of values
        """
        return self.get_mse(_y_test, _y_pred) / np.mean(list(np.transpose(_y_pred)[0]))


class DemandPredictorNN(DemandPredictorBase):
    def __init__(self, _x, _y, train=True):
        self.input_shape = len(_x[0])
        self.output_shape = len(_y[0])
        self.epochs = 4000
        self.batch_size = 150
        self.verbose = 0
        self.validation_split = 0.2
        self.learning_rate = 0.01
        self.history = None
        super(DemandPredictorNN, self).__init__(_x, _y, train)

    def build_model(self):
        """
            build the model
        """
        model = Sequential()
        model.add(Dense(168, input_shape=(self.input_shape,), activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(24, activation='relu'))
        model.add(Dropout(0.01))
        model.add(Dense(self.output_shape, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def predict(self, _x_test):
        if os.path.exists("best_model.h5"):
            self.model = load_model("best_model.h5")
        return super(DemandPredictorNN, self).predict(_x_test)

    def train(self):
        """
            train the model
        """
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=self.verbose, patience=1000)
        mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=self.verbose, save_best_only=True)
        self.history = \
            self.model.fit(
                self.x, self.y,
                epochs=self.epochs, batch_size=self.batch_size,
                verbose=self.verbose, validation_split=self.validation_split,
                use_multiprocessing=True, callbacks=[es, mc]
            )

    def save(self, model_file_name):
        """
        :param model_file_name: file name for the model
        Save the model based on input file name model_file_name given
        """
        model_json = self.model.to_json()
        with open(f"{model_file_name}.json", "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(f"{model_file_name}.h5")

    def load(self, model_file_name):
        with open(f"{model_file_name}", 'r') as json_file:
            loaded_model_json = json_file.read()

        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(f"{model_file_name}.h5")

    def plot(self, _i, _mse=None):
        import matplotlib.pyplot as plt
        losses = self.history.history['loss']
        val_losses = self.history.history['val_loss']
        epochs = [i for i in range(len(losses))]
        plt.xscale('log')
        plt.xlabel("Epochs [Log Scale]")
        plt.ylabel("Loss")
        legend = ["Val Loss", "Train Loss"]
        if _mse is not None:
            plt.plot(epochs, [_mse for _ in range(len(epochs))])
            legend = ["MSE"] + legend
        plt.plot(epochs, val_losses)
        plt.plot(epochs, losses)
        plt.legend(legend, loc="lower center", bbox_to_anchor=(0.5, 0.0))
        plt.title("Variation of Loss function over time")
        plt.savefig(f'sample_{_i}.png')
        plt.close()


class DemandPredictorSVR(DemandPredictorBase):
    def __init__(self, _x, _y, train=True):
        self.kernel = 'rbf'
        self.gamma = 10
        self.c = 10
        super(DemandPredictorSVR, self).__init__(_x, _y, train)

    def build_model(self):
        """
            build the model
        """
        model = SVR(kernel=self.kernel, gamma=self.gamma, C=self.c)
        return model

    def train(self):
        """
            train the model
        """
        self.model.fit(self.x, self.y)

    def get_mse_avg(self, _y_test, _y_pred):
        """
        :param _y_test: actual test values
        :param _y_pred: predicted test values
        :return: return the ratio between MSE and average of values
        """
        return self.get_mse(_y_test, _y_pred) / np.mean(list(np.transpose(_y_pred)))


def transform_data(_processed_data):
    columns = list(_processed_data.columns)
    columns.remove("Number of Trips")
    sc_x = StandardScaler()
    sc_y = MinMaxScaler()
    x = np.array(
        [
            [entry[col] for col in columns]
            for _, entry in _processed_data.iterrows()
        ]
    )
    y = np.transpose([_processed_data["Number of Trips"].to_list()])
    x = sc_x.fit_transform(x)
    y = sc_y.fit_transform(y)
    return x, y, sc_x, sc_y


def solve_using_neural_network(_processed_data):
    """
    :param _processed_data: processed_data
    :return: run NUM_SAMPLES time neural network and compute the average MSE and MSE ratio
    """
    _x, _y, _, _ = transform_data(_processed_data)

    mse = []
    mse_ratio = []

    for _i in range(NUM_SAMPLES):
        x_train, x_test, y_train, y_test = train_test_split(_x, _y, test_size=0.1, random_state=_i)
        tf.random.set_seed(_i)
        nn = DemandPredictorNN(_x=x_train, _y=y_train)
        y_pred = nn.predict(x_test)
        mse_val = nn.get_mse(y_test, y_pred)
        nn.plot(f"sample_{_i}", mse_val)
        mse.append(mse_val)
        mse_ratio.append(nn.get_mse_avg(y_test, y_pred))

    print(f"Average MSE: {np.mean(mse)}, Min MSE: {min(mse)}, Max MSE: {max(mse)}")
    print(f"Average MSE Ratio: {np.mean(mse_ratio)}, Min MSE Ratio: {min(mse_ratio)}, Max MSE Ratio: {max(mse_ratio)}")


def solve_k_fold_neural_network(_processed_data):
    """
    :param _processed_data: processed_data
    :return: run NUM_SAMPLES time neural network and compute the average MSE and MSE ratio
    """
    _x, _y, _, _ = transform_data(_processed_data)
    _i = 0
    mse = []
    mse_ratio = []
    tf.random.set_seed(_i)
    kf = KFold(n_splits=10, random_state=1, shuffle=True)
    for _k, (train_idx, test_idx) in enumerate(kf.split(_x, _y)):
        x_train, y_train = _x[train_idx], _y[train_idx]
        x_test, y_test = _x[test_idx], _y[test_idx]
        nn = DemandPredictorNN(_x=x_train, _y=y_train)
        y_pred = nn.predict(x_test)
        mse_val = nn.get_mse(y_test, y_pred)
        nn.plot(f"sample_{_k}", mse_val)
        mse.append(mse_val)
        mse_ratio.append(nn.get_mse_avg(y_test, y_pred))

    print(f"Average MSE: {np.mean(mse)}, Min MSE: {min(mse)}, Max MSE: {max(mse)}")
    print(f"Average MSE Ratio: {np.mean(mse_ratio)}, Min MSE Ratio: {min(mse_ratio)}, Max MSE Ratio: {max(mse_ratio)}")


def solve_using_svr(_processed_data):
    """
    :param _processed_data: processed_data
    :return: run NUM_SAMPLES time neural network and compute the average MSE and MSE ratio
    """
    _x, _y, _, _ = transform_data(_processed_data)

    mse = []
    mse_ratio = []

    for _i in range(NUM_SAMPLES):
        x_train, x_test, y_train, y_test = train_test_split(_x, _y, test_size=0.1, random_state=_i)
        svr = DemandPredictorSVR(x_train, y_train)
        y_pred = svr.predict(x_test)
        mse.append(svr.get_mse(y_test, y_pred))
        mse_ratio.append(svr.get_mse_avg(y_test, y_pred))

    print(f"Average MSE: {np.mean(mse)}, Min MSE: {min(mse)}, Max MSE: {max(mse)}")
    print(f"Average MSE Ratio: {np.mean(mse_ratio)}, Min MSE Ratio: {min(mse_ratio)}, Max MSE Ratio: {max(mse_ratio)}")


_data, _demand, _demand_w, _demand_h, _demand_wh = load_data("data", data_size=100)
_demand_data = {
    "base": _demand, "weekday": _demand_w,
    "hour": _demand_h, "weekday_n_hour": _demand_wh
}
for _demand_key in _demand_data:
    _demand_datum = _demand_data[_demand_key]
    print(f"checking {_demand_key}")
    solve_using_svr(_demand_w)
    solve_using_neural_network(_demand_w)
