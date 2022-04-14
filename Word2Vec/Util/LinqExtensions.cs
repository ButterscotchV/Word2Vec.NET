namespace Word2Vec.Util
{
    public static class LinqExtensions
    {
        public static IEnumerable<IEnumerable<T>> Partition<T>(this IEnumerable<T> items,
                                                       int partitionSize)
        {
            int i = 0;
            // TODO Check if ToArray() is needed?
            return items.GroupBy(x => i++ / partitionSize);
        }
    }
}
