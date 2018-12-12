using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkSimple
{
    public class Neuron
    {

        /// <summary>
        /// принять сигналы со всех входов с весами и передать результат активационной функции на все выходы
        /// </summary>
        public void SendSignals()
        {

        }
        /// <summary>
        /// активационная функция
        /// </summary>
        public Func<double, double> ActivationFunction { get; }

        /// <summary>
        /// производная активационной функции
        /// </summary>
        public Func<double, double> ActivationDerivative { get; }

        /// <summary>
        /// нейроны следующего слоя, с которыми связан данный нейрон
        /// </summary>
        List<Neuron> NextTierNeurons;

        /// <summary>
        /// результат активационной функции нейрона (удобно отдельно просматривать для нейронов внешнего слоя)
        /// </summary>
        public double Output { get; }

    }
}
