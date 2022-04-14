namespace Word2Vec
{
    public class Word2VecModel<TToken> where TToken : notnull
    {
        public readonly TToken[] Vocab;
        public readonly int LayerSize;
        public readonly double[][] Vectors;

        public Word2VecModel(TToken[] vocab, int layerSize, double[][] vectors)
        {
            Vocab = vocab;
            LayerSize = layerSize;
            Vectors = vectors;
        }

        /// <returns><see cref="Word2VecTrainerBuilder{TToken}"/> for training a model</returns>
        public static Word2VecTrainerBuilder<TToken> Trainer()
        {
            return new();
        }
    }
}
