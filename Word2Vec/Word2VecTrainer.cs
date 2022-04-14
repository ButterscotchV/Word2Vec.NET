using System.Linq;
using Word2Vec.NeuralNetwork;
using Word2Vec.Util;

namespace Word2Vec
{
    internal class Word2VecTrainer
    {
        private readonly int minVocabFrequency;
        private readonly MultiSet<int>? vocab;
        private readonly NeuralNetworkConfig config;

        public Word2VecTrainer(int minVocabFrequency, MultiSet<int>? vocab, NeuralNetworkConfig config)
        {
            this.minVocabFrequency = minVocabFrequency;
            this.vocab = vocab;
            this.config = config;
        }

        private static MultiSet<int> Count(IEnumerable<int> tokens)
        {
            return new(tokens);
        }

        private OrderedMultiSet<int> FilterAndSort(MultiSet<int> counts)
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

            MultiSet<int> counts;
            Console.WriteLine("Acquiring word frequencies");
            listener.Update(TrainingProgressListener.Stage.AcquireVocab, 0.0);
            counts = this.vocab ?? Count(ConcatAll(sentences));

            OrderedMultiSet<int> vocab;
            Console.WriteLine("Filtering and sorting vocabulary");
            listener.Update(TrainingProgressListener.Stage.FilterSortVocab, 0.0);
            vocab = FilterAndSort(counts);

            //Dictionary<int, HuffmanNode> huffmanNodes;

            throw new NotImplementedException();
        }
    }
}
