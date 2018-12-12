using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkSimple
{
    public class NeuralNetwork
    {
        //tiers[i] = количество нейронов в i-м слое
        public NeuralNetwork(List<int> tiers)
        {
            TiersCount = tiers.Count;
            Weights = new List<double[,]>();
            Biases = new List<double[,]>();//new List<List<double>>();
            var rndWeightsGen = new Random();
            /*Biases.Add(new List<double>(tiers[0]));

            for (int i = 0; i < tiers[0]; ++i)
                Biases[0][i] = 0; //входной слой просто считывает данные - смещения нулевые
                */
            for (int i = 0; i < TiersCount - 1; ++i)
            {
                //Biases.Add(new List<double>(tiers[i + 1]));
                Biases.Add(new double[,]);
                for (int j = 0; i < tiers[i + 1]; ++j)
                    Biases[i, j] = rndWeightsGen.NextDouble();
                Weights.Add(new double[tiers[i + 1], tiers[i]]);
                for (int k = 0; i < tiers[i + 1]; ++k)
                    for (int j = 0; j < tiers[i]; ++j)
                        Weights.Last()[k, j] = rndWeightsGen.NextDouble();
            }
        }

        public int TiersCount {get; }

        ///запускаем сеть на заданном входном наборе
        public void RunNetwork()
        {

        }

        /*List<Double> MultiplyVectByMatrix(List<double> vect, double[,] matrix)
        {
            var res = new List<double>();
            double i_th_elem = 0;
            for (int i = 0; i < matrix.GetLength(0); ++i)
            {
                for (int j = 0; j < matrix.GetLength(1); ++j)
                    i_th_elem += vect[j] * matrix[i, j];
                res.Add(i_th_elem);
            }
            return res;
        }*/

        List<Double> SumVectors(List<double> v1, List<double> v2)
        {
            var res = new List<double>();
            for (int i = 0; i < v1.Count; ++i)
                res.Add(v1[i] + v2[i]);
            return res;
        }

        double[,] MultMatrix(double[,] arr1, double[,] arr2)
        {
            var res = new double[arr1.GetLength(0), arr2.GetLength(1)];
            for (int i = 0; i < arr1.GetLength(0); ++i)
                for (int j = 0; j < arr1.GetLength(1); ++j)
                    res[i, j] += arr1[i, j] * arr2[j, i];
            return res;
        }

        double[,] SumMatrix(double[,] arr1, double[,] arr2)
        {
            var res = new double[arr1.GetLength(0), arr2.GetLength(1)];
            for (int i = 0; i < arr1.GetLength(0); ++i)
                for (int j = 0; j < arr1.GetLength(1); ++j)
                    res[i, j] = arr1[i, j] + arr2[i, j];
            return res;
        }


        double[,] Transpose(double[,] arr)
        {
            var res = new double[arr.GetLength(1), arr.GetLength(0)];
            for (int i = 0; i < arr.GetLength(0); ++i)
                for (int j = 0; j < arr.GetLength(1); ++j)
                    res[i, j] = arr[j, i];
            return res;
        }

        /// <summary>
        /// получить активации для слоя l + 1
        /// </summary>
        /// <param name="l"></param>
        /// <param name="l_outputs"></param> вектор данных из слоя l
        /// <returns></returns>
        public List<double> GetNextInputs(int l, List<double> l_outputs)
        {
            var weightedOutputs = MultiplyVectByMatrix(l_outputs, Weights[l]);
            return SumVectors(weightedOutputs, Shifts[l + 1]);
        }

        public List<double> Multiply(double[,] weights, List<double> inputs)
        {
            var res = new List<double>();
            double tmpRes = 0;
            for (int i = 0; i < weights.GetLength(0); ++i)
            {
                for (int j = 0; j < inputs.Count; ++j)
                    tmpRes += weights[i, j] * inputs[j];
                res.Add(tmpRes);
                tmpRes = 0;
            }
            return res;
        }

        public List<Double> GetSigmas(List<double> inputs) => inputs.Select(i => Sigma(i)).ToList();

        public List<Double> GetDerivatives(List<double> inputs) => inputs.Select(i => SigmaDerivative(i)).ToList();

        public static Func<double, double> Sigma = x => 1 / (1 + Math.Exp(-x));

        public static Func<double, double> SigmaDerivative = x => Sigma(x) * (1 - Sigma(x));

        //вычисление в целом: череда вычислений активаций очередного уровня и передача их результатов на следующий уровень
        ///Weights[i] - слой весов из i-1 в i-й слой
        public List<double[,]> Weights;

        ///Векторы смещений слоёв
        public List<double[,]> Biases;
        //public List<List<double>> Biases;

        public List<double> MultiplyVecors(List<double> v1, List<double> v2)
        {
            var res = new List<double>();
            for (int i = 0; i < v1.Count; ++i)
                res.Add(v1[i] * v2[i]);
            return res;
        }

        //public void BackPropagation(List<double> inputs, List<double> ys)
        /*public Tuple<List<double[,]>, List<double[,]>> BackPropagation(double[,] inputs, double[,] ys)
        {
            var gradientB = new List<double[,]>(); //gradients of biases
            var gradientW = new List<double[,]>(); //gradients of weights
            var activation = inputs;
            var activations = new List<double[,]>(); //activations of layers
            activations.Add(activation);
            var zs = new List<double[,]>(); //sums of layers
            for (int i = 0; i < Weights.Count; ++i)
            {
                var b = Biases[i + 1];
                var w = Weights[i];
                var z = SumMatrix(MultMatrix(w, activation), b); //SumVectors(Multiply(w, activation), b);
                zs.Add(z);
                activation = GetSigmas(z);
                activations.Add(activation);
            }

            //var delta = MultiplyVecors(CostDerivative(activations.Last(), ys), GetDerivatives(zs.Last()));
            var delta = MultMatrix(CostDerivative(activations.Last(), ys), GetDerivatives(zs.Last()));
            gradientB.Add(delta);
            gradientW.Add(MultMatrix(delta, Transpose(activations[activations.Count - 2])));

            for (int l = 1; l < activations.Count - 1; ++l)
            {
                var z = zs[zs.Count - 1 - l];
                var sDerivative = GetDerivatives(z);
                delta = MultMatrix(Transpose(Weights[Weights.Count - l]), delta); //TODO: * sDerivative // - 1 - l + 1 = -l
                gradientB.Add(delta);
                gradientW.Add(MultMatrix(delta, Transpose(activations[activations.Count - 2 - l])));
            }

            return new Tuple<List<double[,]>, List<double[,]>>(gradientB, gradientW);
        }*/

        //public Tuple<List<double[,]>, List<double[,]>> BackPropagation(double[,] inputs, double[,] ys)
        public void BackPropagation(List<double> inputs, List<double> ys)
        {
            var gradientB = new List<List<double>>(); //gradients of biases
            var gradientW = new List<double[,]>(); //gradients of weights
            var activation = inputs;
            var activations = new List<List<double>>(); //activations of layers
            activations.Add(activation);
            var zs = new List<List<double>>(); //sums of layers
            for (int i = 0; i < Weights.Count; ++i)
            {
                var b = Biases[i];
                var w = Weights[i];
                var z = Multiply(w, activation);//SumMatrix(MultMatrix(w, activation), b); //SumVectors(Multiply(w, activation), b);
                zs.Add(z);
                activation = GetSigmas(z);
                activations.Add(activation);
            }

            var delta = MultiplyVecors(CostDerivative(activations.Last(), ys), GetDerivatives(zs.Last()));
            //var delta = MultMatrix(CostDerivative(activations.Last(), ys), GetDerivatives(zs.Last()));
            gradientB.Add(delta);
            gradientW.Add(MultMatrix(delta, Transpose(activations[activations.Count - 2])));

            for (int l = 1; l < activations.Count - 1; ++l)
            {
                var z = zs[zs.Count - 1 - l];
                var sDerivative = GetDerivatives(z);
                delta = MultMatrix(Transpose(Weights[Weights.Count - l]), delta); //TODO: * sDerivative // - 1 - l + 1 = -l
                gradientB.Add(delta);
                gradientW.Add(MultMatrix(delta, Transpose(activations[activations.Count - 2 - l])));
            }

            return new Tuple<List<double[,]>, List<double[,]>>(gradientB, gradientW);
        }

        List<Double> CostDerivative(List<double> v1, List<double> v2)
        {
            var res = new List<double>();
            for (int i = 0; i < v1.Count; ++i)
                res.Add(v1[i] - v2[i]);
            return res;
        }

        /*
         *             z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
         * */

        /*double[,] CostDerivative(double[,] outputActivations, double[,] ys)
        {
            var res = new double[outputActivations.GetLength(0), outputActivations.GetLength(1)];
            for (int i = 0; i < outputActivations.GetLength(0); ++i)
                for (int j = 0; j < outputActivations.GetLength(1); ++j)
                    res[i, j] = outputActivations[i, j] - ys[i, j];
            return res;
        }*/
    }

    /*
     * class Network(object):
...
   def backprop(self, x, y):
        """Return a tuple "(nabla_b, nabla_w)" representing the
        gradient for the cost function C_x.  "nabla_b" and
        "nabla_w" are layer-by-layer lists of numpy arrays, similar
        to "self.biases" and "self.weights"."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z)) 
     * 
     * */
}
