import yaml
from yaml import CLoader
import numpy as np
from scipy.sparse import linalg
from pathlib import Path
from tqdm import tqdm


class Base:

    def __init__(self):

        self._species = None
        self._reaction_matrix = None
        self._rates_from_config = None
        self._rates = None
        self._rate_strings = None
        self._reactants = None
        self._products = None

        self._data = self._load_data_from_yaml()
        self._process_data_from_config()

        self._initialize_species_vector()
        self._initialize_reaction_matrix()
        self._initialize_rates()

    def _load_data_from_yaml(self):

        data = {}
        config_file = Path.cwd() / 'reactions.cfg'
        with open(config_file) as file:
            data.update(yaml.load(file, Loader=CLoader))

        return data

    def _process_data_from_config(self):

        reactions = [list(reaction.keys())[0] for reaction in self._data['reactions']]
        reactions = [reaction.replace(' ', '') for reaction in reactions]
        reactants = [reaction.split("->")[0] for reaction in reactions]
        reactants = [reactant.split('+') for reactant in reactants]
        products = [reaction.split("->")[1] for reaction in reactions]
        products = [product.split('+') for product in products]

        self._reactants = reactants
        self._products = products
        self._rates_from_config = self._data['rates']

    def _initialize_species_vector(self):

        reactants = self._reactants
        products = self._products

        all_species = []
        for reactant, product in zip(reactants, products):
            for species in reactant:
                all_species.append(species)
            for species in product:
                all_species.append(species)

        all_species.remove('0')
        all_species = sorted(list(set(all_species)))

        self._species = all_species

    def _initialize_reaction_matrix(self):

        reactants = self._reactants
        products = self._products

        n_reactions = len([list(reaction.keys())[0] for reaction in self._data['reactions']])
        n_species = len(self.species)

        reaction_matrix = np.zeros(shape=(n_reactions,n_species), dtype=int)

        for row, reactant, product in zip(reaction_matrix, reactants, products):
            for i, species in enumerate(self.species):
                if species in reactant:
                    row[i] += -1
                if species in product:
                    row[i] += 1

        self._reaction_matrix = reaction_matrix

    def _initialize_rates(self):

        rates_dictionary = self._rates_from_config
        rate_strings = [list(reaction.values())[0] for reaction in self._data['reactions']]
        rate_values = np.empty(shape=len(rate_strings))

        for i, rate in enumerate(rate_strings):
            rate_values[i] = rates_dictionary[rate]

        self._rates = rate_values
        self._rate_strings = rate_strings

    @property
    def species(self):
        return self._species

    @property
    def reaction_matrix(self):
        return self._reaction_matrix

    @property
    def rates(self):
        return self._rates

    @property
    def rate_strings(self):
        return self._rate_strings

    def run(self):
        pass


class MasterEquation(Base):

    def __init__(self, initial_species=None):
        super(MasterEquation, self).__init__()

        self.initial_state = None
        self.constitutive_states = None
        self.constitutive_states_strings = None
        self.generator_matrix= None
        self.generator_matrix_strings = None
        self.P_t = None

        self._initial_species = initial_species
        if initial_species is not None:
            self._set_initial_state()
            self._set_constitutive_states()
            self._set_generator_matrices()

    def _set_initial_state(self):

        initial_state = np.zeros(shape=(len(self.species)), dtype=np.int32)
        all_species = np.array(self.species)

        for species, quantity in self._initial_species.items():
            if species in self.species:
                i = np.argwhere(all_species == species)
                initial_state[i] = quantity

        self.initial_state = initial_state

    def _set_constitutive_states(self):

        constitutive_states = [list(self.initial_state)]
        newly_added_unique_states = [self.initial_state]
        while True:
            accepted_candidate_states = []
            for state in newly_added_unique_states:
                for reaction in self.reaction_matrix:
                    reactants_required = reaction < 0
                    indices = np.argwhere(reactants_required).transpose()
                    reactants_available = state > 0
                    if np.all(reactants_available[indices]):
                        new_candidate_state = state + reaction
                        if list(new_candidate_state) not in constitutive_states:
                            accepted_candidate_states.append(new_candidate_state)
                            constitutive_states.append(list(new_candidate_state))

            newly_added_unique_states = [state for state in accepted_candidate_states]
            if not newly_added_unique_states:
                break

        constitutive_states_strings = []
        for state in constitutive_states:
            word = []
            for quantity, species in zip(state, self.species):
                #if quantity == 0 and 'E' in species:
                #    pass
                #else:
                word.append(f'{quantity}{species}')

            constitutive_states_strings.append(word)

        self.constitutive_states = constitutive_states
        self.constitutive_states_strings =  constitutive_states_strings

    def _set_generator_matrices(self):

        N = len(self.constitutive_states)

        generator_matrix_values = np.empty(shape=(N,N), dtype=np.int32)
        generator_matrix_strings = []

        for i in range(N):
            new_row = []
            state_i = self.constitutive_states[i]
            for j in range(N):
                state_j = self.constitutive_states[j]

                if i == j:
                    rate_string = '-'
                    rate_value = 0
                for k, reaction in enumerate(self.reaction_matrix):
                    if list(state_i + reaction) == state_j:
                        rate_string = fr'{self.rate_strings[k]}'
                        rate_value = self.rates[k]
                        break
                else:
                    rate_string = '0'
                    rate_value = 0

                new_row.append(rate_string)
                generator_matrix_values[i][j] = rate_value

            generator_matrix_strings.append(new_row)

        for i in range(N):
            generator_matrix_values[i][i] = -np.sum(generator_matrix_values[i])

        self.generator_matrix_strings = generator_matrix_strings
        self.generator_matrix = generator_matrix_values

    def update_rates(self, new_rates):

        for key, value in new_rates.items():
            self._rates_from_config[key] = value
        self._initialize_rates()
        self._set_generator_matrices()

    def run(self, start=None, stop=None, step=None):

        n_steps = int((stop-start) / step)
        dt = step

        N = len(self.constitutive_states)
        P_0 = np.zeros(shape=N)
        P_t = np.empty(shape=(n_steps,N))

        for i, state in enumerate(self.constitutive_states):
            if np.array_equal(state, self.initial_state):
                P_0[i] = 1
                break

        P_t[0] = P_0

        Gt = self.generator_matrix * dt
        propagator = linalg.expm(Gt)

        for i in tqdm(range(n_steps - 1)):
            P_t[i+1] = P_t[i].dot(propagator)

        self.P_t = P_t

    def calculate_mutual_information(self, X, Y, t=-1):

        X = sorted(X)
        Y = sorted(Y)

        indices_X = [self.species.index(x) for x in X]
        indices_Y = [self.species.index(y) for y in Y]

        constitutive_states = np.array(self.constitutive_states)
        x_states = constitutive_states[:, indices_X]
        y_states = constitutive_states[:, indices_Y]
        assert len(x_states) == len(y_states)
        n_states = len(x_states)

        X_set = []
        for vector in x_states:
            if list(vector) not in X_set:
                X_set.append(list(vector))
        X_set = np.array(X_set)

        Y_set = []
        for vector in y_states:
            if list(vector) not in Y_set:
                Y_set.append(list(vector))
        Y_set = np.array(Y_set)

        P = self.P_t[t]
        def _single_term(x, y):
            p_xy = 0
            p_x = 0
            p_y = 0
            for i in range(n_states):
                if np.array_equal(x, x_states[i]) and np.array_equal(y, y_states[i]):
                    p_xy += P[i]
                if np.array_equal(x, x_states[i]):
                    p_x += P[i]
                if np.array_equal(y, y_states[i]):
                    p_y += P[i]

            try:
                MI_term = p_xy * (np.log(p_xy) - np.log(p_x) - np.log(p_y))
            except:
                MI_term = 0

            return MI_term

        mutual_information = 0
        for x in X_set:
            for y in Y_set:
                term = _single_term(x, y)
                mutual_information += term

        return mutual_information


class Gillespie(Base):

    def __init__(self):
        super(Base, self).__init__()

    def run(self):
        pass
