using Word2Vec.Huffman;
using Word2Vec.NeuralNetwork;
using Word2Vec.Util;

namespace Word2Vec
{
    public class Word2VecTrainer<TToken> where TToken : notnull
    {
        private readonly int minVocabFrequency;
        private readonly AbstractMultiSet<TToken>? vocab;
        private readonly NeuralNetworkConfig neuralNetworkConfig;

        public Word2VecTrainer(int minVocabFrequency, AbstractMultiSet<TToken>? vocab, NeuralNetworkConfig neuralNetworkConfig)
        {
            this.minVocabFrequency = minVocabFrequency;
            this.vocab = vocab;
            this.neuralNetworkConfig = neuralNetworkConfig;
        }

        private static MultiSet<TToken> Count(IEnumerable<TToken> tokens)
        {
            return new(tokens);
        }

        private OrderedMultiSet<TToken> FilterAndSort(AbstractMultiSet<TToken> counts)
        {
            OrderedMultiSet<TToken> cleaned = new(counts);

            cleaned.Filter(x => x.Count >= minVocabFrequency);
            cleaned.SortHighestCountFirst();

            return cleaned;
        }

        private IEnumerable<TToken> ConcatAll(IEnumerable<List<TToken>> sentences)
        {
            foreach (var sentence in sentences)
            {
                foreach (var token in sentence)
                {
                    yield return token;
                }
            }
        }

        public Word2VecModel<TToken> Train(TrainingProgressListener listener, IEnumerable<List<TToken>> sentences)
        {
            // TODO Add timers?

            AbstractMultiSet<TToken> counts;
            Console.WriteLine("Acquiring word frequencies");
            listener.Update(TrainingProgressListener.Stage.AcquireVocab, 0.0);
            counts = this.vocab ?? Count(ConcatAll(sentences));

            OrderedMultiSet<TToken> vocab;
            Console.WriteLine("Filtering and sorting vocabulary");
            listener.Update(TrainingProgressListener.Stage.FilterSortVocab, 0.0);
            vocab = FilterAndSort(counts);

            Dictionary<TToken, HuffmanCoding<TToken>.HuffmanNode> huffmanNodes;
            Console.WriteLine("Create Huffman encoding");
            huffmanNodes = new HuffmanCoding<TToken>(vocab, listener).Encode();

            NeuralNetworkTrainer<TToken>.NeuralNetworkModel model;
            Console.WriteLine($"Training model {neuralNetworkConfig}");
            model = neuralNetworkConfig.CreateTrainer(vocab, huffmanNodes, listener).Train(sentences);

            return new Word2VecModel<TToken>(vocab.ElementSet(), model.LayerSize, model.Vectors);
        }
    }
}
