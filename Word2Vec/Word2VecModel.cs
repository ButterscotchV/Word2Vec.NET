namespace Word2Vec
{
    public class Word2VecModel
    {
        public readonly List<int> Vocab;
        public readonly int LayerSize;
        public readonly List<double[]> Vectors;

        public Word2VecModel(List<int> vocab, int layerSize, List<double[]> vectors)
        {
            Vocab = vocab;
            LayerSize = layerSize;
            Vectors = vectors;
        }
    }
}
