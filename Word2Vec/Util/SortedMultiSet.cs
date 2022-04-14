using System.Collections;

namespace Word2Vec.Util
{
    public class SortedMultiSet<T> : AbstractMultiSet<T>, IEnumerable<T>, ICloneable where T : notnull
    {
        private readonly SortedDictionary<T, int> internalDict = new();
        public override IDictionary<T, int> Dict => internalDict;

        public SortedMultiSet() : base()
        {
        }

        public SortedMultiSet(IEnumerable<T> items) : base(items)
        {
        }

        public SortedMultiSet(AbstractMultiSet<T> multiSet) : base(multiSet)
        {
        }

        private SortedMultiSet(SortedDictionary<T, int> internalDict)
        {
            this.internalDict = internalDict;
        }

        public override SortedMultiSet<T> Clone()
        {
            SortedMultiSet<T> clone = new(new SortedDictionary<T, int>(internalDict));
            return clone;
        }
    }
}
