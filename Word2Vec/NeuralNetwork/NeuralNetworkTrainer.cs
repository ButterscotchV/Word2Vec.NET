using Word2Vec.Huffman;
using Word2Vec.Util;

namespace Word2Vec.NeuralNetwork
{
    // Parent class for training word2vec's neural network
    public abstract class NeuralNetworkTrainer<TToken> where TToken : notnull
    {
        /// <summary>
        /// Sentences longer than this are broken into multiple chunks
        /// </summary>
        private static readonly int MaxSentenceLength = 1_000;

        /// <summary>
        /// Boundary for maximum exponent allowed
        /// </summary>
        public static readonly int MaxExp = 6;

        /// <summary>
        /// Size of the pre-cached exponent table
        /// </summary>
        public static readonly int ExpTableSize = 1_000;
        public static readonly double[] ExpTable = new double[ExpTableSize];

        static NeuralNetworkTrainer()
        {
            for (int i = 0; i < ExpTableSize; i++)
            {
                // Precompute the Exp() table
                ExpTable[i] = Math.Exp((i / (double)ExpTableSize * 2 - 1) * MaxExp);
                // Precompute f(x) = x / (x + 1)
                ExpTable[i] /= ExpTable[i] + 1;
            }
        }

        private static readonly int TableSize = (int) 1e8;

        private readonly TrainingProgressListener listener;

        public readonly NeuralNetworkConfig config;
        public readonly Dictionary<TToken, HuffmanCoding<TToken>.HuffmanNode> huffmanNodes;
        private readonly int vocabSize;
        public readonly int layer1_size;
        public readonly int window;

        /// <summary>
        /// In the C version, this includes the </s> token that replaces a newline character
        /// </summary>
        public long numTrainedTokens;

        // The following includes shared state that is updated per worker thread

        // USE INTERLOCKED!!
        /// <summary>
        /// To be precise, this is the number of words in the training data that exist in the vocabulary
        /// which have been processed so far. It includes words that are discarded from sampling.
        /// Note that each word is processed once per iteration.
        /// </summary>
        protected int actualWordCount = 0;

        /// <summary>
        /// Learning rate, affects how fast values in the layers get updated
        /// </summary>
        public double alpha;

        /// <summary>
        /// This contains the outer layers of the neural network.
        /// First dimension is the vocab, second is the layer.
        /// </summary>
        public readonly double[][] syn0;

        /// <summary>
        /// This contains hidden layers of the neural network
        /// </summary>
        public readonly double[][] syn1;

        /// <summary>
        /// This is used for negative sampling
        /// </summary>
        private readonly double[][] syn1neg;

        /// <summary>
        /// Used for negative sampling
        /// </summary>
        private readonly int[] table;
        public long startNano;

        public NeuralNetworkTrainer(NeuralNetworkConfig config, OrderedMultiSet<TToken> vocab, Dictionary<TToken, HuffmanCoding<TToken>.HuffmanNode> huffmanNodes, TrainingProgressListener listener)
        {
            this.config = config;
            this.huffmanNodes = huffmanNodes;
            this.listener = listener;
            this.vocabSize = huffmanNodes.Count;
            this.numTrainedTokens = vocab.ElementCount;
            this.layer1_size = config.layerSize;
            this.window = config.windowSize;

            this.alpha = config.initialLearningRate;

            this.syn0 = new double[vocabSize][];
            this.syn1 = new double[vocabSize][];
            this.syn1neg = new double[vocabSize][];

            // Recreate [vocabSize][layer1_size]
            for (var i = 0; i < vocabSize; i++)
            {
                this.syn0[i] = new double[layer1_size];
                this.syn1[i] = new double[layer1_size];
                this.syn1neg[i] = new double[layer1_size];
            }

            this.table = new int[TableSize];

            InitializeSyn0();
            InitializeUnigramTable();
        }

        private void InitializeUnigramTable()
        {
            long trainWordsPow = 0;
            double power = 0.75;

            foreach (HuffmanCoding<TToken>.HuffmanNode node in huffmanNodes.Values)
            {
                trainWordsPow += (long)Math.Pow(node.Count, power);
            }

            var nodeIter = huffmanNodes.Values.GetEnumerator();
            if (!nodeIter.MoveNext())
            {
                throw new Exception("Failed to initialize unigram table, there are no huffman nodes");
            }

            HuffmanCoding<TToken>.HuffmanNode last = nodeIter.Current;
            double d1 = Math.Pow(last.Count, power) / trainWordsPow;
            int i = 0;
            for (int a = 0; a < TableSize; a++)
            {
                table[a] = i;
                if (a / (double)TableSize > d1)
                {
                    i++;
                    HuffmanCoding<TToken>.HuffmanNode next = nodeIter.MoveNext() ? nodeIter.Current : last;

                    d1 += Math.Pow(next.Count, power) / trainWordsPow;

                    last = next;
                }
            }
        }

        private void InitializeSyn0()
        {
            long nextRandom = 1;
            for (int a = 0; a < huffmanNodes.Count; a++)
            {
                // Consume a random for fun
                // Actually we do this to use up the injected </s> token
                nextRandom = IncrementRandom(nextRandom);
                for (int b = 0; b < layer1_size; b++)
                {
                    nextRandom = IncrementRandom(nextRandom);
                    syn0[a][b] = (((nextRandom & 0xFFFF) / (double)65_536) - 0.5) / layer1_size;
                }
            }
        }

        /// <returns>Next random value to use</returns>
        public static long IncrementRandom(long r)
        {
            return r * 25_214_903_917L + 11;
        }

        /// <summary>
        /// Represents a neural network model
        /// </summary>
        public interface NeuralNetworkModel
        {
            // Size of the layers
            public int LayerSize { get; }

            // Resulting vectors
            public double[][] Vectors { get; }
        }

        public class BasicNeuralNetworkModel : NeuralNetworkModel
        {
            private readonly int layerSize;
            private readonly double[][] vectors;

            // Size of the layers
            public int LayerSize => layerSize;
            // Resulting vectors
            public double[][] Vectors => vectors;

            public BasicNeuralNetworkModel(int layerSize, double[][] vectors)
            {
                this.layerSize = layerSize;
                this.vectors = vectors;
            }
        }

        public NeuralNetworkModel Train(IEnumerable<List<TToken>> sentences)
        {
            // Create an executor that runs as many threads as are defined in the config, and blocks if
            // you're trying to run more. This is to make sure we don't read the entire corpus into
            // memory.
            // TODO Add executor

            int numSentences = sentences.Count();
            numTrainedTokens += numSentences;

            var batched = sentences.Partition(1024);

            try
            {
                listener.Update(TrainingProgressListener.Stage.TrainNeuralNetwork, 0.0);
                for (int iter = config.iterations; iter > 0; iter--)
                {
                    // Some list of Tasks?...
                    List<Task> tasks = new(64);
                    int i = 0;
                    foreach (var batch in batched)
                    {
                        tasks.Add(Task.Run(() => CreateWorker(i, iter, batch).Run()));
                        i++;
                    }

                    try
                    {
                        Task.WhenAll(tasks).Wait();
                    }
                    catch (ExecutionException e)
                    {
                        throw new InvalidOperationException("Error training neural network", e);
                    }
                }
            }
            finally
            {
                // TODO Add shutdown?
            }

            return new BasicNeuralNetworkModel(config.layerSize, syn0);
        }

        /// <returns><see cref="Worker"/> to process the given sentences</returns>
        protected abstract Worker CreateWorker(int randomSeed, int iter, IEnumerable<List<TToken>> batch);

        /// <summary>
        /// Worker thread that updates the neural network model
        /// </summary>
        public abstract class Worker
        {
            protected static readonly int LearningRateUpdateFrequency = 10_000;

            public long NextRandom;
            public readonly int Iter;
            public readonly IEnumerable<List<TToken>> Batch;

            /// <summary>
            /// The number of words observed in the training data for this worker that exist
            /// in the vocabulary.It includes words that are discarded from sampling.
            /// </summary>
            public int WordCount;

            /// <summary>
            /// Value of wordCount the last time alpha was updated
            /// </summary>
            public int LastWordCount;

            public readonly double[] neu1;
            public readonly double[] neu1e;

            protected NeuralNetworkTrainer<TToken> networkTrainer;

            public Worker(int randomSeed, int iter, IEnumerable<List<TToken>> batch, NeuralNetworkTrainer<TToken> networkTrainer)
            {
                this.NextRandom = randomSeed;
                this.Iter = iter;
                this.Batch = batch;

                neu1 = new double[networkTrainer.layer1_size];
                neu1e = new double[networkTrainer.layer1_size];

                this.networkTrainer = networkTrainer;
            }

            public void Run()
            {
                foreach (var sentence in Batch)
                {
                    List<TToken> filteredSentence = new(sentence.Count);
                    foreach (var s in sentence)
                    {
                        if (!networkTrainer.huffmanNodes.ContainsKey(s))
                        {
                            continue;
                        }

                        WordCount++;
                        if (networkTrainer.config.downSampleRate > 0)
                        {
                            var huffmanNode = networkTrainer.huffmanNodes[s];
                            double random = (Math.Sqrt(huffmanNode.Count / (networkTrainer.config.downSampleRate * networkTrainer.numTrainedTokens)) + 1)
                                * (networkTrainer.config.downSampleRate * networkTrainer.numTrainedTokens) / huffmanNode.Count;
                            NextRandom = IncrementRandom(NextRandom);
                            if (random < (NextRandom & 0xFFFF) / (double)65_536)
                            {
                                continue;
                            }
                        }

                        filteredSentence.Add(s);
                    }

                    // Increment word count one extra for the injected </s> token
                    // Turns out if you don't do this, the produced word vectors aren't as tasty
                    WordCount++;

                    var partitioned = filteredSentence.Partition(MaxSentenceLength);
                    foreach (var chunked in partitioned)
                    {
                        // TODO Handle thread interrupt

                        if (WordCount - LastWordCount > LearningRateUpdateFrequency)
                        {
                            UpdateAlpha(Iter);
                        }
                        TrainSentence(chunked.ToList());
                    }
                }

                Interlocked.Add(ref networkTrainer.actualWordCount, WordCount - LastWordCount);
            }

            /// <summary>
            /// Degrades the learning rate (alpha) steadily towards 0
            /// </summary>
            /// <param name="iter">Only used for debugging</param>
            private void UpdateAlpha(int iter)
            {
                int currentActual = Interlocked.Add(ref networkTrainer.actualWordCount, WordCount - LastWordCount);
                LastWordCount = WordCount;

                // Degrade the learning rate linearly towards 0 but keep a minimum
                networkTrainer.alpha = networkTrainer.config.initialLearningRate * Math.Max(1 - currentActual / (double)(networkTrainer.config.iterations * networkTrainer.numTrainedTokens), 0.0001);

                networkTrainer.listener.Update(TrainingProgressListener.Stage.TrainNeuralNetwork, currentActual / (double)(networkTrainer.config.iterations * networkTrainer.numTrainedTokens + 1));
            }

            public void HandleNegativeSampling(HuffmanCoding<TToken>.HuffmanNode huffmanNode)
            {
                for (int d = 0; d <= networkTrainer.config.negativeSamples; d++)
                {
                    int target;
                    int label;
                    if (d == 0)
                    {
                        target = huffmanNode.Idx;
                        label = 1;
                    }
                    else
                    {
                        NextRandom = IncrementRandom(NextRandom);
                        target = networkTrainer.table[(int)(((NextRandom >> 16) % TableSize) + TableSize) % TableSize];
                        if (target == 0)
                            target = (int)(((NextRandom % (networkTrainer.vocabSize - 1)) + networkTrainer.vocabSize - 1) % (networkTrainer.vocabSize - 1)) + 1;
                        if (target == huffmanNode.Idx)
                            continue;
                        label = 0;
                    }
                    int l2 = target;
                    double f = 0;
                    for (int c = 0; c < networkTrainer.layer1_size; c++)
                        f += neu1[c] * networkTrainer.syn1neg[l2][c];
                    double g;
                    if (f > MaxExp)
                        g = (label - 1) * networkTrainer.alpha;
                    else if (f < -MaxExp)
                        g = (label) * networkTrainer.alpha;
                    else
                        g = (label - ExpTable[(int)((f + MaxExp) * (ExpTableSize / MaxExp / 2))]) * networkTrainer.alpha;
                    for (int c = 0; c < networkTrainer.layer1_size; c++)
                        neu1e[c] += g * networkTrainer.syn1neg[l2][c];
                    for (int c = 0; c < networkTrainer.layer1_size; c++)
                        networkTrainer.syn1neg[l2][c] += g * neu1[c];
                }
            }

            /// <summary>
            /// Update the model with the given raw sentence
            /// </summary>
            public abstract void TrainSentence(List<TToken> unfiltered);
        }
    }
}
