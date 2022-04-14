using Word2Vec.NeuralNetwork;
using Word2Vec.Util;

namespace Word2Vec
{
    public class Word2VecTrainerBuilder<TToken> where TToken : notnull
    {
        private int layerSize = 100;
        private int windowSize = 5;
        private int numThreads = Environment.ProcessorCount;
        private int negativeSamples;
        private int minVocabFrequency = 5;
        private double? initialLearningRate = null;
        private double downSampleRate = 0.001;
        private int iterations = 5;

        public int LayerSize
        {
            get => layerSize;

            set
            {
                if (value <= 0)
                {
                    throw new ArgumentOutOfRangeException(nameof(value), "Value must be positive");
                }
                layerSize = value;
            }
        }

        public int WindowSize
        {
            get => windowSize;

            set
            {
                if (value <= 0)
                {
                    throw new ArgumentOutOfRangeException(nameof(value), "Value must be positive");
                }
                windowSize = value;
            }
        }

        public int NumThreads
        {
            get => numThreads;

            set
            {
                if (value <= 0)
                {
                    throw new ArgumentOutOfRangeException(nameof(value), "Value must be positive");
                }
                numThreads = value;
            }
        }

        public NeuralNetworkTypeValues.NeuralNetworkType Type { get; set; } = NeuralNetworkTypeValues.CBOW;

        public int NegativeSamples
        {
            get => negativeSamples;

            set
            {
                if (value < 0)
                {
                    throw new ArgumentOutOfRangeException(nameof(value), "Value must be non-negative");
                }
                negativeSamples = value;
            }
        }

        public bool UseHierarchicalSoftmax { get; set; } = false;

        public AbstractMultiSet<TToken>? Vocab { get; set; } = null;

        public int MinVocabFrequency
        {
            get => minVocabFrequency;
            
            set
            {
                if (value < 0)
                {
                    throw new ArgumentOutOfRangeException(nameof(value), "Value must be non-negative");
                }
                minVocabFrequency = value;
            }
        }

        public double? InitialLearningRate
        {
            get => initialLearningRate;
            
            set
            {
                if (value < 0)
                {
                    throw new ArgumentOutOfRangeException(nameof(value), "Value must be non-negative");
                }
                initialLearningRate = value;
            }
        }

        public double DownSampleRate
        {
            get => downSampleRate;
            
            set
            {
                if (value < 0)
                {
                    throw new ArgumentOutOfRangeException(nameof(value), "Value must be non-negative");
                }
                downSampleRate = value;
            }
        }

        public int Iterations
        {
            get => iterations;
            
            set
            {
                if (value <= 0)
                {
                    throw new ArgumentOutOfRangeException(nameof(value), "Value must be positive");
                }
                iterations = value;
            }
        }

        public TrainingProgressListener? Listener { get; set; } = null;

        public Word2VecModel<TToken> Train(IEnumerable<List<TToken>> sentences)
        {
            double inititalLearningRate = InitialLearningRate ?? Type.DefaultInitialLearningRate;
            TrainingProgressListener listener = Listener ?? new();

            NeuralNetworkConfig config = new NeuralNetworkConfig(Type, NumThreads, Iterations, LayerSize, WindowSize, NegativeSamples, DownSampleRate, inititalLearningRate, UseHierarchicalSoftmax);
            return new Word2VecTrainer<TToken>(MinVocabFrequency, Vocab, config).Train(listener, sentences);
        }
    }

    public class TrainingProgressListener
    {
        public enum Stage
        {
            AcquireVocab,
            FilterSortVocab,
            CreateHuffmanEncoding,
            TrainNeuralNetwork,
        }

        public virtual void Update(Stage stage, double progress)
        {
            Console.WriteLine($"Stage {Enum.GetName(stage)}, progress {progress*100.0:0.00}%");
        }
    }
}
