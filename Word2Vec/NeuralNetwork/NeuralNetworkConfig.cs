namespace Word2Vec.NeuralNetwork
{
    public class NeuralNetworkConfig
    {
        public readonly NeuralNetworkTypeValues.NeuralNetworkType type;
        public readonly int numThreads;
        public readonly int iterations;
        public readonly int layerSize;
        public readonly int windowSize;
        public readonly int negativeSamples;
        public readonly double downSampleRate;
        public readonly double initialLearningRate;
        public readonly bool useHierarchicalSoftmax;

        // TODO Implement this
        /*
        public NeuralNetworkTrainer CreateTrainer(List<int> vocab, Dictionary<int, HuffmanNode> huffmanNodes, TrainingProgressListener listener)
        {
            return type.CreateTrainer(this, vocab, huffmanNodes, listener);
        }
        */

        public NeuralNetworkConfig(NeuralNetworkTypeValues.NeuralNetworkType type, int numThreads, int iterations, int layerSize, int windowSize, int negativeSamples, double downSampleRate, double initialLearningRate, bool useHierarchicalSoftmax)
        {
            this.type = type;
            this.numThreads = numThreads;
            this.iterations = iterations;
            this.layerSize = layerSize;
            this.windowSize = windowSize;
            this.negativeSamples = negativeSamples;
            this.downSampleRate = downSampleRate;
            this.initialLearningRate = initialLearningRate;
            this.useHierarchicalSoftmax = useHierarchicalSoftmax;
        }
    }
}
