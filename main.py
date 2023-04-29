import csv
import datetime
import io
import os
from os.path import exists
from subprocess import Popen, PIPE

import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

data_set_address: str = "C:/Files/Local File/DataSET.csv"
labeled_address: str = "C:/Files/Local File/Labeled.csv"
demo_address: str = "C:/Files/Local File/demo.csv"
raw_line_address: str = "C:/Files/Local File/raw.txt"
error_address: str = "C:/Files/Local File/Error.txt"
test_address: str = "C:/Files/Local File/test.csv"
log_path: str = "C:/Files/ssl_access_log"
file_path: str = "C:/Files/Local File"

log_files = os.listdir(log_path)


# Prepare raw data to training
class Preparation:
    source_name: str
    source: io
    data_set: io
    demo_set: io
    raw_set: io
    days = {2}
    days.remove(2)

    def prepare(self):

        exist = self.filtering()

        if not exist:
            return

        self.parsing()
        group, days = self.grouping()
        repeat_in_time = self.counting(group, days)
        self.saving(repeat_in_time, days)

    def filtering(self):

        exist = False

        self.raw_set = open(raw_line_address, "a", newline='')

        for source_name in log_files:
            self.source = open((log_path + "/" + source_name), "r")

            print("the File in path: {dir}, filter...".format(dir=source_name))

            for line in self.source:
                index = line.find("group-statement-balance")
                if index != -1:
                    if line[index:].find("\" 200 ") != -1:
                        self.raw_set.write(line)

            print("the File in path: {dir}, filtered!".format(dir=source_name))

            exist = True

            self.source.close()

        self.raw_set.close()

        return exist

    def parsing(self):
        file_r = open(raw_line_address, "r", newline="")
        file_exist = exists(demo_address)
        file_w = open(demo_address, "a", newline="")

        fieldnames = ['Day', 'Month', 'Year', 'Hour', 'Minute']
        writer = csv.DictWriter(file_w, fieldnames=fieldnames)
        if not file_exist:
            writer.writeheader()

        print("the File in path: demo.csv, pars...")

        for line in file_r:
            data = line.split("[")[1]
            data = data.split("]")[0]
            data = data.split(":")
            hour = data[1]
            minute = data[2]
            data = data[0]
            data = data.split("/")
            day = data[0]
            self.days.add(int(day))
            month = data[1]
            year = data[2]

            writer.writerow({'Day': day.__str__(),
                             'Month': month.__str__(),
                             'Year': year.__str__(),
                             'Hour': hour.__str__(),
                             'Minute': minute.__str__()})

        print("the File in path: demo.csv, parsed!")

        file_r.close()
        open(raw_line_address, "w").close()
        file_w.close()

    @staticmethod
    def grouping():
        dataframe = pd.read_csv(demo_address)

        print("the data, group...")

        group = dataframe.groupby(["Day", "Hour", "Minute"])
        days = dataframe.groupby(["Day"])

        print("the data, grouped!")

        return group, days

    @staticmethod
    def counting(group, days):

        repeat_in_time = {}

        print("the group, count...")

        for day in days:
            for hour in range(0, 24):
                for minute in range(0, 60):

                    key1 = (day[0] + (hour + (minute + 2) // 60) // 24).__str__()
                    key2 = ((hour + (minute + 2) // 60) % 24).__str__()
                    key3 = ((minute + (minute % 2) + (2 * ((minute + 1) % 2))) % 60).__str__()

                    try:
                        if group.groups.get((day[0], hour, minute)) is not None:
                            repeat_in_time[(key1 + "-" + key2 + ":" + key3)] += len(
                                group.groups.get((day[0], hour, minute)))
                    except:
                        repeat_in_time[(key1 + "-" + key2 + ":" + key3)] = len(
                            group.groups.get((day[0], hour, minute)))

        print("the group, counted!")

        return repeat_in_time

    @staticmethod
    def saving(repeat_in_time, days):
        dataframe = pd.read_csv(demo_address)
        file_w = open(file_path + "/" + "DataSET.csv", "a", newline="")
        writer = csv.writer(file_w)

        print("Preparing DataSET.csv!")
        year = dataframe.get('Year')[0]
        month_num = 0
        match dataframe.get('Month')[0]:
            case 'Jan':
                month_num = 1
            case '2':
                month_num = 2
            case '3':
                month_num = 3
            case '4':
                month_num = 4
            case '5':
                month_num = 5
            case '6':
                month_num = 6
            case 'Jul':
                month_num = 7
            case 'Aug':
                month_num = 8
            case 'Sep':
                month_num = 9
            case 'Oct':
                month_num = 10
            case 'Nov':
                month_num = 11
            case 'Dec':
                month_num = 12

        for day in days:
            for hour in range(0, 24):
                for minute in range(0, 60, 2):

                    key1 = (day[0] + (hour + (minute + 2) // 60) // 24).__str__()
                    key2 = ((hour + (minute + 2) // 60) % 24).__str__()
                    key3 = ((minute + (minute % 2) + (2 * ((minute + 1) % 2))) % 60).__str__()

                    try:
                        writer.writerow([day[0],
                                         month_num,
                                         year,
                                         hour,
                                         minute,
                                         repeat_in_time[(key1 + "-" + key2 + ":" + key3)]])
                    except:
                        pass

        print("The DataSET.csv is ready")

        demo_rewrite = open(demo_address, "w")
        fieldnames = ['Day', 'Month', 'Year', 'Hour', 'Minute']
        writer = csv.DictWriter(demo_rewrite, fieldnames=fieldnames)
        writer.writeheader()
        demo_rewrite.close()
        file_w.close()


# training the machine
class Training:
    data_set: io
    dataframe: pd.DataFrame

    def openfile(self):
        self.data_set = open(data_set_address, "r")
        # names = ['Date', 'Time', 'URL', 'Status code', "Number", "Tag"]
        names = ['day', 'month', 'year', 'Hour', 'Minute', "Number", "Tag"]
        self.dataframe = pd.read_csv(data_set_address, names=names)

    def train(self):
        self.openfile()

        print(self.dataframe.describe())

        array = self.dataframe.values
        x = array[:, 0:-1]
        y = array[:, -1]
        x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.20, random_state=1)

        # Spot Check Algorithms
        # evaluate each model in turn
        models = [
            ('LDA ', LinearDiscriminantAnalysis()),
            ('KNN ', KNeighborsClassifier()),
            ('CART', DecisionTreeClassifier()),
            ('MNB ', MultinomialNB()),
            ('GNB ', GaussianNB())]
        results = []
        names = []

        max_result = 0
        max_model = []

        print("Training has begun!\n")

        for name, ai_model in models:
            kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
            cv_results = cross_val_score(ai_model, x_train, y_train, cv=kfold, scoring='accuracy')
            if cv_results.mean() > max_result:
                max_result = cv_results.mean()
                max_model = [ai_model, name]
            results.append(cv_results)
            names.append(name)
            print('%s: mean = %f (std = %f)' % (name, cv_results.mean(), cv_results.std()))

        # Make predictions on validation dataset
        my_model = max_model[0]
        my_model.fit(x_train, y_train)
        predictions = my_model.predict(x_validation)

        # Evaluate predictions
        print('data accuracy of ' + max_model[1] + ': ' + accuracy_score(y_validation, predictions).__str__())

        print("\nTraining has been completed!")

        self.data_set.close()
        return my_model


# use machine to decide
class Judgment:

    def counting(self, group, days):

        repeat_in_time = {}

        test_dataframe = pd.read_csv(test_address)
        minute_set = test_dataframe.get("Minute")

        print("the group, count...")

        for day in days:
            for hour in range(0, 24):
                for minute in range(0, 60):

                    key1 = (day[0] + (hour + (minute + 2) // 60) // 24).__str__()
                    key2 = ((hour + (minute + 2) // 60) % 24).__str__()
                    key3 = ((minute + (minute % 2) + (2 * ((minute + 1) % 2))) % 60).__str__()

                    try:
                        if group.groups.get((day[0], hour, minute)) is not None:
                            repeat_in_time[(key1 + "-" + key2 + ":" + key3)] += len(
                                group.groups.get((day[0], hour, minute)))
                    except:
                        repeat_in_time[(key1 + "-" + key2 + ":" + key3)] = len(
                            group.groups.get((day[0], hour, minute)))
                        repeat_in_time[(key1 + "-" + key2 + ":" + key3)] += self.exists_num(minute_set, test_dataframe)

        print("the group, counted!")
        return repeat_in_time

    @staticmethod
    def saving(days, repeat_in_time):
        dataframe = pd.read_csv(demo_address)
        file_w = open(file_path + "/" + "examine.csv", "w", newline="")
        writer = csv.writer(file_w)

        print("Preparing examine.csv!")

        year = dataframe.get('Year')[0]
        month_num = 0
        match dataframe.get('Month')[0]:
            case 'Jan':
                month_num = 1
            case '2':
                month_num = 2
            case '3':
                month_num = 3
            case '4':
                month_num = 4
            case '5':
                month_num = 5
            case '6':
                month_num = 6
            case 'Jul':
                month_num = 7
            case 'Aug':
                month_num = 8
            case 'Sep':
                month_num = 9
            case 'Oct':
                month_num = 10
            case 'Nov':
                month_num = 11
            case 'Dec':
                month_num = 12

        for day in days:
            for hour in range(0, 24):
                for minute in range(0, 60, 2):

                    key1 = (day[0] + (hour + (minute + 2) // 60) // 24).__str__()
                    key2 = ((hour + (minute + 2) // 60) % 24).__str__()
                    key3 = ((minute + (minute % 2) + (2 * ((minute + 1) % 2))) % 60).__str__()

                    try:
                        writer.writerow([day[0],
                                         month_num,
                                         year,
                                         hour,
                                         minute,
                                         repeat_in_time[(key1 + "-" + key2 + ":" + key3)]])
                    except:
                        pass

        print("The examine.csv is ready")

        demo_rewrite = open(demo_address, "w")
        fieldnames = ['Day', 'Month', 'Hour', 'Minute']
        writer = csv.DictWriter(demo_rewrite, fieldnames=fieldnames)
        writer.writeheader()
        demo_rewrite.close()
        file_w.close()

    @staticmethod
    def distance(minute, data):
        if minute - data > 2 or minute - data > -58:
            return True
        return False

    def judge(self, ai_model):

        exist = Preparation().filtering()
        if exist:
            Preparation().parsing()
            group, days = Preparation().grouping()

            repeat_in_time = self.counting(group, days)
            self.saving(days, repeat_in_time)

        array = self.normalize()
        if len(array) > 0:
            predictions = ai_model.predict(array)

            self.error_writing(predictions, array)

    @staticmethod
    def exists_num(value, dataframe):
        index = 0
        try:
            for x in value:

                if x == 3:
                    return dataframe.get("number")[index]

                index += 1
        except TypeError:
            pass
        return 0

    @staticmethod
    def normalize():
        dataframe = pd.read_csv(file_path + "/normal.csv")
        normal_value = dataframe.values
        names = ['day', 'month', 'year', 'hour', 'minute', "number"]
        dataframe = pd.read_csv(file_path + "/examine.csv", names=names)
        open(file_path + "/examine.csv", "w", newline="").close()
        array = dataframe.values
        for i in range(0, len(array)):
            array[i][7] = 100 * array[i][7] // normal_value[(array[i][5] * 30) + (array[i][6] // 2)][7]

        return array

    @staticmethod
    def error_writing(predictions, array):
        file_error = open(error_address, "a", newline="")
        index = 0
        for pred in predictions:
            if pred == "invalid":
                timeform = "[" + \
                           datetime.datetime.now().strftime("%d") + "/" + \
                           datetime.datetime.now().strftime("%B") + "/" + \
                           datetime.datetime.now().strftime("%Y") + ":" + \
                           datetime.datetime.now().hour.__str__() + ":" + \
                           datetime.datetime.now().minute.__str__() + ":" + \
                           datetime.datetime.now().second.__str__() + " +4:30] --> "
                file_error.write(timeform)
                file_error.write(int(array[index][5]).__str__() + ":" + int(array[index][6]).__str__())
                file_error.write("\n")

            index += 1


def run(command):
    process = Popen(command, stdout=PIPE, shell=True)
    writer = csv.writer(open((log_path + "/logs.log"), "a"))

    while True:
        line = process.stdout.readline().rstrip()
        if not line:
            Judgment().judge(model)
        writer.writerow(line)


if __name__ == "__main__":
    # Preparation().prepare()
    train = Training()
    model = train.train()

    cmd = """for (( i = 2; i >=0; i-- )) ; do
        grep $(date +%R -d "-$i  min") /var/log/httpd/ssl_access_log
           done"""

    run(cmd)
