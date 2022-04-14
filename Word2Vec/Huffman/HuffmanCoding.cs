using Word2Vec.Util;

namespace Word2Vec.Huffman
{
    public class HuffmanCoding<TToken> where TToken : notnull
    {
        // Node
        public class HuffmanNode
        {
            // Array of 0's and 1's
            public readonly byte[] Code;

            // Array of parent node index offsets
            public readonly int[] Point;

            // Index of the Huffman node
            public readonly int Idx;

            // Frequency of the token
            public readonly int Count;

            public HuffmanNode(byte[] code, int[] point, int idx, int count)
            {
                Code = code;
                Point = point;
                Idx = idx;
                Count = count;
            }
        }

        private readonly OrderedMultiSet<TToken> vocab;
        private readonly TrainingProgressListener listener;

        public HuffmanCoding(OrderedMultiSet<TToken> vocab, TrainingProgressListener listener)
        {
            this.vocab = vocab;
            this.listener = listener;
        }

        public Dictionary<TToken, HuffmanNode> Encode()
        {
            int numTokens = vocab.EntryCount;

            int[] parentNode = new int[numTokens * 2 + 1];
            byte[] binary = new byte[numTokens * 2 + 1];
            long[] count = new long[numTokens * 2 + 1];

            int i = 0;
            foreach (var entry in vocab.GetAsEntryEnumerable())
            {
                count[i] = entry.Count;
                i++;
            }

            if (i != numTokens) {
                throw new Exception($"Expected {i} to match {numTokens}");
            }

            for (i = numTokens; i < count.Length; i++)
            {
                count[i] = (long) 1e15;
            }

            CreateTree(numTokens, count, binary, parentNode);

            return Encode(binary, parentNode);
        }

        /*
         * Populate the count, binary, and parentNode arrays with the Huffman tree
         * This uses the linear time method assuming that the count array is sorted
         */
        private void CreateTree(int numTokens, long[] count, byte[] binary, int[] parentNode)
        {
            int min1i;
            int min2i;
            int pos1 = numTokens - 1;
            int pos2 = numTokens;

            // Construct the Huffman tree by adding one node at a time
            for (int a = 0; a < numTokens - 1; a++)
            {
                // First, find two smallest nodes 'min1, min2'
                if (pos1 >= 0)
                {
                    if (count[pos1] < count[pos2])
                    {
                        min1i = pos1;
                        pos1--;
                    }
                    else
                    {
                        min1i = pos2;
                        pos2++;
                    }
                }
                else
                {
                    min1i = pos2;
                    pos2++;
                }

                if (pos1 >= 0)
                {
                    if (count[pos1] < count[pos2])
                    {
                        min2i = pos1;
                        pos1--;
                    }
                    else
                    {
                        min2i = pos2;
                        pos2++;
                    }
                }
                else
                {
                    min2i = pos2;
                    pos2++;
                }

                int newNodeIdx = numTokens + a;
                count[newNodeIdx] = count[min1i] + count[min2i];
                parentNode[min1i] = newNodeIdx;
                parentNode[min2i] = newNodeIdx;
                binary[min2i] = 1;

                if (a % 1_000 == 0)
                {
                    // TODO Add process interrupt?
                    listener.Update(TrainingProgressListener.Stage.CreateHuffmanEncoding, (0.5 * a) / numTokens);
                }
            }
        }

        private Dictionary<TToken, HuffmanNode> Encode(byte[] binary, int[] parentNode)
        {
            int numTokens = vocab.EntryCount;

            // Now assign binary code to each unique token
            Dictionary<TToken, HuffmanNode> result = new();
            int nodeIdx = 0;
            foreach (var entry in vocab.GetAsEntryEnumerable())
            {
                int curNodeIdx = nodeIdx;
                List<byte> code = new();
                List<int> points = new();

                while (true)
                {
                    code.Add(binary[curNodeIdx]);
                    points.Add(curNodeIdx);
                    curNodeIdx = parentNode[curNodeIdx];
                    if (curNodeIdx == numTokens * 2 - 2)
                        break;
                }

                int codeLen = code.Count;
                int count = entry.Count;
                byte[] rawCode = new byte[codeLen];
                int[] rawPoints = new int[codeLen + 1];

                rawPoints[0] = numTokens - 2;
                for (int i = 0; i < codeLen; i++)
                {
                    rawCode[codeLen - i - 1] = code[i];
                    rawPoints[codeLen - i] = points[i] - numTokens;
                }

                TToken token = entry.Element;
                result.Add(token, new HuffmanNode(rawCode, rawPoints, nodeIdx, count));

                if (nodeIdx % 1_000 == 0)
                {
                    // TODO Add process interrupt?
                    listener.Update(TrainingProgressListener.Stage.CreateHuffmanEncoding, 0.5 + (0.5 * nodeIdx) / numTokens);
                }

                nodeIdx++;
            }

            return result;
        }
    }
}
