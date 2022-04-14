namespace Word2Vec
{
    public class Word2VecModel
    {
        public readonly int[] Vocab;
        public readonly int LayerSize;
        public readonly double[][] Vectors;

        public Word2VecModel(int[] vocab, int layerSize, double[][] vectors)
        {
            Vocab = vocab;
            LayerSize = layerSize;
            Vectors = vectors;
        }
    }
}
