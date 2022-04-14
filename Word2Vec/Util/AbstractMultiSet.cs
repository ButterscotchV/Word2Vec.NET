using System.Collections;

namespace Word2Vec.Util
{
    public abstract class AbstractMultiSet<T> : IEnumerable<T>, IEnumerable<MultiSetEntry<T>>, ICloneable where T : notnull
    {
        public abstract IDictionary<T, int> Dict { get; }

        public AbstractMultiSet()
        {
        }

        public AbstractMultiSet(IEnumerable<T> items)
        {
            Add(items);
        }

        public AbstractMultiSet(AbstractMultiSet<T> multiSet)
        {
            foreach (var item in multiSet.Dict)
            {
                Dict.Add(item.Key, item.Value);
            }
        }

        public int Count(T item)
        {
            return Dict.ContainsKey(item) ? Dict[item] : 0;
        }

        public bool Contains(T item)
        {
            return Dict.ContainsKey(item);
        }

        public void Add(T item)
        {
            if (Dict.ContainsKey(item))
                Dict[item]++;
            else
                Dict[item] = 1;
        }

        public void Add(IEnumerable<T> items)
        {
            foreach (var item in items)
                Add(item);
        }

        public void Remove(T item)
        {
            if (!Dict.ContainsKey(item))
                throw new ArgumentException($"{nameof(item)} is not contained", nameof(item));
            if (--Dict[item] == 0)
                Dict.Remove(item);
        }

        // Return the last value in the multiset
        public T Peek()
        {
            if (!Dict.Any())
                throw new NullReferenceException();
            return Dict.Last().Key;
        }

        // Return the last value in the multiset and remove it.
        public T Pop()
        {
            T item = Peek();
            Remove(item);
            return item;
        }

        public IEnumerator<T> GetEnumerator()
        {
            foreach (var kvp in Dict)
                for (int i = 0; i < kvp.Value; i++)
                    yield return kvp.Key;
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public IEnumerator<MultiSetEntry<T>> GetEntryEnumerator()
        {
            foreach (var kvp in Dict)
                yield return new MultiSetEntry<T>(kvp);
        }

        IEnumerator<MultiSetEntry<T>> IEnumerable<MultiSetEntry<T>>.GetEnumerator()
        {
            return GetEntryEnumerator();
        }

        public void Filter(Predicate<MultiSetEntry<T>> predicate)
        {
            foreach (var entry in (IEnumerable<MultiSetEntry<T>>)this)
            {
                if (!predicate(entry))
                {
                    Dict.Remove(entry.Entry);
                }
            }
        }

        public abstract AbstractMultiSet<T> Clone();

        object ICloneable.Clone()
        {
            return Clone();
        }
    }
}
