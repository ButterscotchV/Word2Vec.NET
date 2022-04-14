using Word2Vec.Huffman;
using Word2Vec.Util;

namespace Word2Vec.NeuralNetwork
{
    public class SkipGramModelTrainer<TToken> : NeuralNetworkTrainer<TToken> where TToken : notnull
    {
        public SkipGramModelTrainer(NeuralNetworkConfig config, OrderedMultiSet<TToken> vocab, Dictionary<TToken, HuffmanCoding<TToken>.HuffmanNode> huffmanNodes, TrainingProgressListener listener) : base(config, vocab, huffmanNodes, listener)
        {
        }

        private class SkipGramWorker : Worker
        {
            public SkipGramWorker(int randomSeed, int iter, IEnumerable<List<TToken>> batch, NeuralNetworkTrainer<TToken> networkTrainer) : base(randomSeed, iter, batch, networkTrainer)
            {
            }

            public override void TrainSentence(List<TToken> sentence)
            {
                int sentenceLength = sentence.Count;

                for (int sentencePosition = 0; sentencePosition < sentenceLength; sentencePosition++)
                {
                    TToken word = sentence[sentencePosition];
                    HuffmanCoding<TToken>.HuffmanNode huffmanNode = networkTrainer.huffmanNodes[word];

                    for (int c = 0; c < networkTrainer.layer1_size; c++)
                        neu1[c] = 0;
                    for (int c = 0; c < networkTrainer.layer1_size; c++)
                        neu1e[c] = 0;
                    NextRandom = IncrementRandom(NextRandom);

                    int b = (int)(((NextRandom % networkTrainer.window) + NextRandom) % networkTrainer.window);

                    for (int a = b; a < networkTrainer.window * 2 + 1 - b; a++)
                    {
                        if (a == networkTrainer.window)
                            continue;
                        int c = sentencePosition - networkTrainer.window + a;

                        if (c < 0 || c >= sentenceLength)
                            continue;
                        for (int d = 0; d < networkTrainer.layer1_size; d++)
                            neu1e[d] = 0;

                        int l1 = networkTrainer.huffmanNodes[sentence[c]].Idx;

                        if (networkTrainer.config.useHierarchicalSoftmax)
                        {
                            for (int d = 0; d < huffmanNode.Code.Length; d++)
                            {
                                double f = 0;
                                int l2 = huffmanNode.Point[d];
                                // Propagate hidden -> output
                                for (int e = 0; e < networkTrainer.layer1_size; e++)
                                    f += networkTrainer.syn0[l1][e] * networkTrainer.syn1[l2][e];

                                if (f <= -MaxExp || f >= MaxExp)
                                    continue;
                                else
                                    f = ExpTable[(int)((f + MaxExp) * (ExpTableSize / MaxExp / 2))];
                                // 'g' is the gradient multiplied by the learning rate
                                double g = (1 - huffmanNode.Code[d] - f) * networkTrainer.alpha;

                                // Propagate errors output -> hidden
                                for (int e = 0; e < networkTrainer.layer1_size; e++)
                                    neu1e[e] += g * networkTrainer.syn1[l2][e];
                                // Learn weights hidden -> output
                                for (int e = 0; e < networkTrainer.layer1_size; e++)
                                    networkTrainer.syn1[l2][e] += g * networkTrainer.syn0[l1][e];
                            }
                        }

                        HandleNegativeSampling(huffmanNode);

                        // Learn weights input -> hidden
                        for (int d = 0; d < networkTrainer.layer1_size; d++)
                        {
                            networkTrainer.syn0[l1][d] += neu1e[d];
                        }
                    }
                }
            }
        }

        protected override Worker CreateWorker(int randomSeed, int iter, IEnumerable<List<TToken>> batch)
        {
            return new SkipGramWorker(randomSeed, iter, batch, this);
        }
    }
}
