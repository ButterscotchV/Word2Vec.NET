using Word2Vec.Huffman;
using Word2Vec.Util;

namespace Word2Vec.NeuralNetwork
{
    public class CBOWModelTrainer : NeuralNetworkTrainer
    {
        public CBOWModelTrainer(NeuralNetworkConfig config, OrderedMultiSet<int> vocab, Dictionary<int, HuffmanCoding.HuffmanNode> huffmanNodes, TrainingProgressListener listener) : base(config, vocab, huffmanNodes, listener)
        {
        }

        private class CBOWWorker : Worker
        {
            public CBOWWorker(int randomSeed, int iter, IEnumerable<List<int>> batch, NeuralNetworkTrainer networkTrainer) : base(randomSeed, iter, batch, networkTrainer)
            {
            }

            public override void TrainSentence(List<int> sentence)
            {
                int sentenceLength = sentence.Count;

                for (int sentencePosition = 0; sentencePosition < sentenceLength; sentencePosition++)
                {
                    int word = sentence[sentencePosition];
                    HuffmanCoding.HuffmanNode huffmanNode = networkTrainer.huffmanNodes[word];

                    for (int c = 0; c < networkTrainer.layer1_size; c++)
                        neu1[c] = 0;
                    for (int c = 0; c < networkTrainer.layer1_size; c++)
                        neu1e[c] = 0;

                    NextRandom = IncrementRandom(NextRandom);
                    int b = (int)((NextRandom % networkTrainer.window) + networkTrainer.window) % networkTrainer.window;

                    // in -> hidden
                    int cw = 0;
                    for (int a = b; a < networkTrainer.window * 2 + 1 - b; a++)
                    {
                        if (a == networkTrainer.window)
                            continue;
                        int c = sentencePosition - networkTrainer.window + a;
                        if (c < 0 || c >= sentenceLength)
                            continue;
                        int idx = networkTrainer.huffmanNodes[sentence[c]].Idx;
                        for (int d = 0; d < networkTrainer.layer1_size; d++)
                        {
                            neu1[d] += networkTrainer.syn0[idx][d];
                        }

                        cw++;
                    }

                    if (cw == 0)
                        continue;

                    for (int c = 0; c < networkTrainer.layer1_size; c++)
                        neu1[c] /= cw;

                    if (networkTrainer.config.useHierarchicalSoftmax)
                    {
                        for (int d = 0; d < huffmanNode.Code.Length; d++)
                        {
                            double f = 0;
                            int l2 = huffmanNode.Point[d];
                            // Propagate hidden -> output
                            for (int c = 0; c < networkTrainer.layer1_size; c++)
                                f += neu1[c] * networkTrainer.syn1[l2][c];
                            if (f <= -MaxExp || f >= MaxExp)
                                continue;
                            else
                                f = ExpTable[(int)((f + MaxExp) * (ExpTableSize / MaxExp / 2))];
                            // 'g' is the gradient multiplied by the learning rate
                            double g = (1 - huffmanNode.Code[d] - f) * networkTrainer.alpha;
                            // Propagate errors output -> hidden
                            for (int c = 0; c < networkTrainer.layer1_size; c++)
                                neu1e[c] += g * networkTrainer.syn1[l2][c];
                            // Learn weights hidden -> output
                            for (int c = 0; c < networkTrainer.layer1_size; c++)
                                networkTrainer.syn1[l2][c] += g * neu1[c];
                        }
                    }

                    HandleNegativeSampling(huffmanNode);

                    // hidden -> in
                    for (int a = b; a < networkTrainer.window * 2 + 1 - b; a++)
                    {
                        if (a == networkTrainer.window)
                            continue;
                        int c = sentencePosition - networkTrainer.window + a;
                        if (c < 0 || c >= sentenceLength)
                            continue;
                        int idx = networkTrainer.huffmanNodes[sentence[c]].Idx;
                        for (int d = 0; d < networkTrainer.layer1_size; d++)
                            networkTrainer.syn0[idx][d] += neu1e[d];
                    }
                }
            }
        }

        protected override Worker CreateWorker(int randomSeed, int iter, IEnumerable<List<int>> batch)
        {
            return new CBOWWorker(randomSeed, iter, batch, this);
        }
    }
}
