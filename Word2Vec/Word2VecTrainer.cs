using Word2Vec.Huffman;
using Word2Vec.NeuralNetwork;
using Word2Vec.Util;

namespace Word2Vec
{
    internal class Word2VecTrainer
    {
        private readonly int minVocabFrequency;
        private readonly AbstractMultiSet<int>? vocab;
        private readonly NeuralNetworkConfig neuralNetworkConfig;

        public Word2VecTrainer(int minVocabFrequency, AbstractMultiSet<int>? vocab, NeuralNetworkConfig neuralNetworkConfig)
        {
            this.minVocabFrequency = minVocabFrequency;
            this.vocab = vocab;
            this.neuralNetworkConfig = neuralNetworkConfig;
        }

        private static MultiSet<int> Count(IEnumerable<int> tokens)
        {
            return new(tokens);
        }

        private OrderedMultiSet<int> FilterAndSort(AbstractMultiSet<int> counts)
        {
            OrderedMultiSet<int> cleaned = new(counts);

            cleaned.Filter(x => x.Count >= minVocabFrequency);
            cleaned.SortHighestCountFirst();

            return cleaned;
        }

        private IEnumerable<int> ConcatAll(IEnumerable<List<int>> sentences)
        {
            foreach (var sentence in sentences)
            {
                foreach (var token in sentence)
                {
                    yield return token;
                }
            }
        }

        public Word2VecModel Train(TrainingProgressListener listener, IEnumerable<List<int>> sentences)
        {
            // TODO Add timers?

            AbstractMultiSet<int> counts;
            Console.WriteLine("Acquiring word frequencies");
            listener.Update(TrainingProgressListener.Stage.AcquireVocab, 0.0);
            counts = this.vocab ?? Count(ConcatAll(sentences));

            OrderedMultiSet<int> vocab;
            Console.WriteLine("Filtering and sorting vocabulary");
            listener.Update(TrainingProgressListener.Stage.FilterSortVocab, 0.0);
            vocab = FilterAndSort(counts);

            Dictionary<int, HuffmanCoding.HuffmanNode> huffmanNodes;
            Console.WriteLine("Create Huffman encoding");
            huffmanNodes = new HuffmanCoding(vocab, listener).Encode();

            NeuralNetworkTrainer.NeuralNetworkModel model;
            Console.WriteLine($"Training model {neuralNetworkConfig}");
            model = neuralNetworkConfig.CreateTrainer(vocab, huffmanNodes, listener).Train(sentences);

            // return new Word2VecModel(vocab.ElementSet(), model.LayerSize, Doubles.Concat(model.Vectors()));
            throw new NotImplementedException();
        }
    }
}
