namespace Word2Vec.Util
{
    public struct MultiSetEntry<TEntry>
    {
        private readonly TEntry entry;
        private readonly int count;

        public MultiSetEntry(TEntry entry, int count)
        {
            this.entry = entry;
            this.count = count;
        }

        public MultiSetEntry(KeyValuePair<TEntry, int> pair) : this(pair.Key, pair.Value)
        {
        }

        public TEntry Entry => entry;

        public int Count => count;

        public override string ToString()
        {
            return $"[{Entry}, {Count}]";
        }
    }
}
