namespace Word2Vec.Util
{
    public class OrderedMultiSet<T> : AbstractMultiSet<T>, IEnumerable<T>, ICloneable where T : notnull
    {
        private readonly OrderedDictionary<T, int> internalDict = new();
        public override IDictionary<T, int> Dict => internalDict;

        public OrderedMultiSet() : base()
        {
        }

        public OrderedMultiSet(IEnumerable<T> items) : base(items)
        {
        }

        public OrderedMultiSet(AbstractMultiSet<T> multiSet) : base(multiSet)
        {
        }

        private OrderedMultiSet(OrderedDictionary<T, int> internalDict)
        {
            this.internalDict = internalDict;
        }

        public override OrderedMultiSet<T> Clone()
        {
            OrderedMultiSet<T> clone = new(new OrderedDictionary<T, int>(internalDict));
            return clone;
        }

        public void SortByCount(Comparison<int> comparison)
        {
            internalDict.SortValues(comparison);
        }

        public void SortHighestCountFirst()
        {
            internalDict.SortValues((x, y) => y.CompareTo(x));
        }
    }
}
