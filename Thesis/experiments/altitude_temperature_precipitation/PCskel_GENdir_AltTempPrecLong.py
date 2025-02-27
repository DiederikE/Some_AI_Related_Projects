import argparse
import random
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from causallearn.search.ConstraintBased.PC import pc
import pydot

'''
--------------------------------
Opmerkingen:
--------------------------------

- Deze code gebruikt het PC algoritme om de causaliteit tussen de variabelen te bepalen,
    hierbij worden de richtingen weggelaten. Dus enkel het skeleton wordt gebruikt.
    
- Vervolgens wordt de richting bepaald aan de hand van de methode met generalisatie.
    Hierbij is het belangrijk dat de juiste splitsing in distributies wordt gemaakt bij elk paar! 

'''

class Model(object):
    def __init__(self, args):
        self.args = args

    def get_model(self):
        return LinearRegression()

    def prepare_data(self, data):
        X, Y = list(zip(*data))
        Y = list(zip(*Y))
        return X, Y

    def train(self, data):
        X, Y = self.prepare_data(data)
        #         X :   shape (n_samples, n_features) -> tuple 175 (arr(float))
        #         y :   shape (n_samples,) or (n_samples, n_targets) -> tuple 175 (float)
        self.reg = [self.get_model().fit(X, y) for y in Y]
        # reg = list of 1 LinearRegression object

    def test(self, data):
        X, Y = self.prepare_data(data)
        return np.mean([1 - reg.score(X, y) for reg, y in zip(self.reg, Y)])


class NNModel(Model):
    def get_model(self):
        hidden_layer_sizes = [100] * self.args.hidden_layers
        return MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                            max_iter=1000)


def separate_distributions(data):
    thresh = len(data) // 2
    left, right = data[thresh:], data[:thresh]

    random.shuffle(left)
    random.shuffle(right)

    thresh_left = len(left) // 10
    left_1, left_2 = left[thresh_left:], left[:thresh_left]
    thresh_right = len(right) // 10
    right_1, right_2 = right[thresh_right:], right[:thresh_right]

    data_train = left_1 + right_2
    data_transfer = right_1 + left_2
    return data_train, data_transfer


def get_generalization_loss(args, data_train, data_transfer):
    if args.hidden_layers == 0:
        model = Model(args)
    else:
        model = NNModel(args)
    model.train(data_train)
    loss_train = model.test(data_train)
    loss_transfer = model.test(data_transfer)
    generalization_loss = loss_transfer - loss_train
    return generalization_loss


def add_noise(data, noise_std):
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = np.random.normal(data[i][j], noise_std)
    return data


def experiment(args, data, noise_std):
    # prepare data
    data_train_AB, data_transfer_AB = separate_distributions(data)
    data_train_AB = np.asarray(data_train_AB)
    data_transfer_AB = np.asarray(data_transfer_AB)
    if noise_std > 0:
        data_train_AB = add_noise(data_train_AB, noise_std)
        data_transfer_AB = add_noise(data_transfer_AB, noise_std)
    data_train_BA = [[x[1], x[0]] for x in data_train_AB]
    data_transfer_BA = [[x[1], x[0]] for x in data_transfer_AB]

    # compute losses and results
    gen_loss_AB = get_generalization_loss(args, data_train_AB,
                                          data_transfer_AB)
    gen_loss_BA = get_generalization_loss(args, data_train_BA,
                                          data_transfer_BA)
    score = gen_loss_BA - gen_loss_AB
    result = score > 0
    return 1 if result else 0

def read_float(x):
    if x == 'NaN':
        return 0
    return float(x)


def get_special_datasets(lines, file_name):
    data = [[[read_float(x)] for x in line.strip().split(' ')] for line in
            lines if len(line.strip()) > 0]

    number_vars = 0
    if len(data) > 0:
        number_vars = len(data[0])

    return data, number_vars


def load_data(args):
    with open(args.file_name, 'r') as f:
        lines = f.readlines()
    # Data is list of lists that represent variables
    data, number_vars = get_special_datasets(lines, args.file_name)

    data = np.asarray(data)
    return data.tolist(), number_vars

def main(args):
    # load data
    data, number_vars = load_data(args)

    pc_data = np.asarray(data)
    pc_data = np.squeeze(pc_data, axis=2)
    cg = pc(pc_data)

    # Draw pc output graph
    # cg.draw_pydot_graph(labels=['Alt', 'Temp', 'Prec', 'Long'])

    graph = pydot.Dot(graph_type='digraph')

    for i in range(number_vars):
        for j in range(i):
            print('Checking relation between:', i, j)

            left_right_value = cg.G.graph[i][j]
            right_left_value = cg.G.graph[j][i]

            if ((left_right_value == -1 and right_left_value == -1)
                  or (left_right_value == 1 and right_left_value == 1)
                  or (left_right_value == -1 and right_left_value == 1)
                  or (left_right_value == 1 and right_left_value == -1)):
                if i == 3 or j == 3:
                    random.shuffle(data)
                    data.sort(key=lambda x: x[3][0])  # TODO: Determines distribution
                else:
                    random.shuffle(data)
                    data.sort(key=lambda x: x[0][0])  # TODO: Determines distribution

                extracted_data = [[data_2d[i], data_2d[j]] for data_2d in data]

                dir_bin = experiment(args, extracted_data, args.noise)
                if dir_bin == 1:
                    print(i, '-->', j)
                    edge = pydot.Edge(map_index_to_value(i), map_index_to_value(j))
                    graph.add_edge(edge)
                else:
                    print(i, '<--', j)
                    edge = pydot.Edge(map_index_to_value(j), map_index_to_value(i))
                    graph.add_edge(edge)
            else:
                print(i, '   ', j)

    graph.write_png('img/pcskel_gendir_alt_temp_prec_long.png')


def map_index_to_value(index):
    if index == 0:
        return 'Alt'
    if index == 1:
        return 'Temp'
    if index == 2:
        return 'Prec'
    if index == 3:
        return 'Long'
    return 'Unknown: ' + str(index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str,
                        default='data/alt_temp_prec_long.txt',
                        help='input file')
    parser.add_argument('--experiments', type=int, default=1000,
                        help='number of experiments')
    parser.add_argument('--original_scale', action='store_true', default=False,
                        help='scale data')
    parser.add_argument('--scale', type=float, default=1,
                        help='scaling')
    parser.add_argument('--noise', type=float, default=0,
                        help='std of noise')
    parser.add_argument('--hidden_layers', type=int, default=0,
                        help='number of hidden layers')
    args = parser.parse_args()
    main(args)
