import numpy
import scipy.special

# 신경망 클래스 정의
class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes    # 입력노드 개수
        self.hnodes = hiddennodes   # 은닉노드 개수
        self.onodes = outputnodes   # 출력노드 개수

        self.lr = learningrate      # 학습률

        # 가중치 행렬(wih, who)
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5),
                                       (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5),
                                       (self.onodes, self.hnodes))

        # self.activation_function = self.sigmoid
        # self.activation_function = self.logistic
        self.activation_function = lambda x : scipy.special.expit(x)
        pass

    def sigmoid(self, x):
        return scipy.special.expit(x)

    def logistic(self, x):
        return 1/(1 + 2.71828 ** (-x))

    def train(self, inputs_list, targets_list):
        "신경망을 학습하는 함수"
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # 순전파 과정을 통해 역전파 학습을 위한 데이터를 리턴받는다
        hidden_outputs, final_outputs = self.query(inputs_list)

        # 출력층 에러와 중간층 에러값을 얻는다
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # 가중치 업데이트 수식을 적용한다(오류 역전파를 통해 가중치 증가/감소 : 학습)
        self.who += self.lr * numpy.dot(
            (output_errors * final_outputs * (1.0 - final_outputs)),
            numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot(
            (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
            numpy.transpose(inputs))

    def query(self, inputs_list):
        # "순전파/질의하기"

        # 입력리스트를 2차원 행렬로 변환
        inputs = numpy.array(inputs_list, ndmin=2).T
        # input -> hidden weight 내적계산
        hidden_inputs = numpy.dot(self.wih, inputs)
        # sigmoid 함수를 통과시킨다
        hidden_outputs = self.activation_function(hidden_inputs)
        # hidden -> output weight 내적계산
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # sigmoid 함수를 통과시킨다
        final_outputs = self.activation_function(final_inputs)

        return hidden_outputs, final_outputs

if __name__ == '__main__':
    input_nodes = 3
    hidden_nodes = 3
    output_nodes = 3
    learning_rate = 0.3

    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    foutputs = n.query([1.0, 0.5, -1.5])
    print(foutputs)