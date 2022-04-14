namespace Word2Vec.Util
{
    public class MultiSet<T> : AbstractMultiSet<T>, IEnumerable<T>, ICloneable where T : notnull
    {
        private readonly Dictionary<T, int> internalDict = new();
        public override IDictionary<T, int> Dict => internalDict;

        public MultiSet() : base()
        {
        }

        public MultiSet(IEnumerable<T> items) : base(items)
        {
        }

        public MultiSet(AbstractMultiSet<T> multiSet) : base(multiSet)
        {
        }

        private MultiSet(Dictionary<T, int> internalDict)
        {
            this.internalDict = internalDict;
        }

        public override MultiSet<T> Clone()
        {
            MultiSet<T> clone = new(new Dictionary<T, int>(internalDict));
            return clone;
        }
    }
}
