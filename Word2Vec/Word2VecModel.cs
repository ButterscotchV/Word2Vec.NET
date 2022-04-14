namespace Word2Vec
{
    public class Word2VecModel<TToken> where TToken : notnull
    {
        public readonly TToken[] Vocab;
        public readonly int LayerSize;
        public readonly double[][] Vectors;

        public Word2VecModel(TToken[] vocab, int layerSize, double[][] vectors)
        {
            if (vocab.Length != vectors.Length)
            {
                throw new ArgumentException($"{nameof(vocab)} [{vocab.Length}] and {nameof(vectors)} [{vectors.Length}] must be the same length!");
            }

            if (vectors.Length > 0 && vectors[0].Length != layerSize)
            {
                throw new ArgumentException($"{nameof(vectors)} [{vectors.Length}] and {nameof(layerSize)} ({layerSize}) must be the same size!");
            }

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
