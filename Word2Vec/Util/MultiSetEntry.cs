namespace Word2Vec.Util
{
    public struct MultiSetEntry<T>
    {
        private readonly T element;
        private readonly int count;

        public MultiSetEntry(T element, int count)
        {
            this.element = element;
            this.count = count;
        }

        public MultiSetEntry(KeyValuePair<T, int> pair) : this(pair.Key, pair.Value)
        {
        }

        public T Element => element;

        public int Count => count;

        public override string ToString()
        {
            return $"[{Element}, {Count}]";
        }
    }
}
