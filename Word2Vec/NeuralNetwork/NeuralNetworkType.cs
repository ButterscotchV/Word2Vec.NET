namespace Word2Vec.NeuralNetwork
{
    public static class NeuralNetworkTypeValues
    {
        public enum NeuralNetworkTypeEnum
        {
            CBOW,
            SKIP_GRAM
        }

        public class NeuralNetworkType
        {
            public readonly NeuralNetworkTypeEnum EnumValue;
            public readonly double DefaultInitialLearningRate;

            public NeuralNetworkType(NeuralNetworkTypeEnum enumValue, double defaultInitialLearningRate)
            {
                EnumValue = enumValue;
                DefaultInitialLearningRate = defaultInitialLearningRate;

                // Set up list
                Values.Add(this);
            }

            // TODO Implement this
            /*
            public NeuralNetworkTrainer CreateTrainer(NeuralNetworkConfig config, List<int> counts, Dictionary<int, HuffmanNode> huffmanNodes, TrainingProgressListener listener)
            {
                switch (EnumValue)
                {
                    case NeuralNetworkTypeEnum.CBOW:
                        return new CBOWModelTrainer(config, counts, huffmanNodes, listener);
                    case NeuralNetworkTypeEnum.SKIP_GRAM:
                        return new SkipGramModelTrainer(config, counts, huffmanNodes, listener);
                    default:
                        throw new NotImplementedException("The requested training option has not been implemented yet");
                }
            }
            */
        }

        public static readonly List<NeuralNetworkType> Values = new();

        public static readonly NeuralNetworkType CBOW = new(NeuralNetworkTypeEnum.CBOW, 0.05);
        public static readonly NeuralNetworkType SKIP_GRAM = new(NeuralNetworkTypeEnum.SKIP_GRAM, 0.025);
        public static NeuralNetworkType? GetByEnumValue(NeuralNetworkTypeEnum enumValue)
        {
            foreach (NeuralNetworkType neuralNetworkType in Values)
            {
                if (neuralNetworkType.EnumValue == enumValue)
                {
                    return neuralNetworkType;
                }
            }

            return null;
        }
    }
}
