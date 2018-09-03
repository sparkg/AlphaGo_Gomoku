import keras
from keras import backend as K
from keras.layers import Conv2D,Dense,Input, Flatten
from keras.regularizers import l2
from keras.optimizers import Adam
import numpy as np
import pickle
class PolicyValueNet():
    """policy-value network """
    def __init__(self, board_width, board_height, model_file=None):
        self.board_width = board_width
        self.board_height = board_height
        self.lr = 5e-3
        self.create_policy_value_net()
        self._loss_train_op()

        if model_file:
            net_params = pickle.load(open(model_file), "rb")
            self.model.set_weights(net_params)

    def loss(self,y_true, y_pred):
        """
        Three loss terms：
        loss = (z - v)^2 + pi^T * log(p) + c||theta||^2
        """
        mse = K.mean(K.square(y_pred - y_true), axis=-1)
        categorical_crossentropy = K.categorical_crossentropy(y_true, y_pred)
        return mse + categorical_crossentropy

    def create_policy_value_net(self):
        """create the policy value network """
        input_ = Input(shape=[self.board_width,self.board_height,4])
        # conv layers
        conv1 = Conv2D(filters=32,kernel_size=[3, 3],padding='same',kernel_regularizer=l2(1e-4),bias_regularizer=l2(1e-4))(input_)
        conv2 = Conv2D(filters=64, kernel_size=[3, 3], padding='same',kernel_regularizer=l2(1e-4),bias_regularizer=l2(1e-4))(conv1)
        conv3 = Conv2D(filters=128, kernel_size=[3, 3], padding='same',kernel_regularizer=l2(1e-4),bias_regularizer=l2(1e-4))(conv2)
        # action policy layers
        policy_net = Conv2D(filters=4, kernel_size=[1, 1],kernel_regularizer=l2(1e-4),bias_regularizer=l2(1e-4))(conv3)
        policy_net = Flatten()(policy_net)
        self.policy_net = Dense(units=self.board_height * self.board_width,activation='softmax',kernel_regularizer=l2(1e-4),bias_regularizer=l2(1e-4))(policy_net)
        # state value layers
        value_net = Conv2D(filters=2, kernel_size=[1,1],kernel_regularizer=l2(1e-4),bias_regularizer=l2(1e-4))(conv3)
        value_net = Flatten()(value_net)
        value_net = Dense(units=64,kernel_regularizer=l2(1e-4),bias_regularizer=l2(1e-4))(value_net)
        self.value_net = Dense(units=1, activation='tanh',kernel_regularizer=l2(1e-4),bias_regularizer=l2(1e-4))(value_net)
        #model
        self.model = keras.Model(inputs=input_,outputs=[self.policy_net,self.value_net])

        def policy_value(state_input):
            state_input_group = np.array(state_input)
            result = self.model.predict_on_batch(state_input_group)
            return result

        self.policy_value = policy_value

    def policy_value(self,input_board):
        return self.model.predict(input_board)

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available action and the score of the board state
        """
        legal_positions = board.availables
        current_state = board.current_state()
        act_probs, value = self.policy_value(current_state.reshape(-1,self.board_width, self.board_height,4))
        act_probs = zip(legal_positions, act_probs.flatten()[legal_positions])
        return act_probs, value[0][0]

    def _loss_train_op(self):
        """
        Three loss terms：
        loss = (z - v)^2 + pi^T * log(p) + c||theta||^2
        """
        # get the train op
        opt = Adam()
        losses = ['categorical_crossentropy', 'mean_squared_error']
        self.model.compile(optimizer=opt, loss=losses)

        def self_entropy(probs):
            return -np.mean(np.sum(probs * np.log(probs + 1e-10), axis=1))

        def train_step(state_input, mcts_probs, winner, learning_rate):
            state_input_union = np.array(state_input)
            mcts_probs_union = np.array(mcts_probs)
            winner_union = np.array(winner)
            loss = self.model.evaluate(state_input_union, [mcts_probs_union, winner_union], batch_size=len(state_input),
                                       verbose=0)
            action_probs, _ = self.model.predict_on_batch(state_input_union)
            entropy = self_entropy(action_probs)
            K.set_value(self.model.optimizer.lr, learning_rate)
            self.model.fit(state_input_union, [mcts_probs_union, winner_union], batch_size=len(state_input), verbose=0)
            return loss[0], entropy

        self.train_step = train_step

    def get_policy_param(self):
        net_params = self.model.get_weights()
        return net_params
    def save_model(self, model):
        net_params = self.get_policy_param()
        pickle.dump(net_params, open(model, "wb"), protocol=2)
