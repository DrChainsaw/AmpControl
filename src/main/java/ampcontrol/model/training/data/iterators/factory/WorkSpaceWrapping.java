package ampcontrol.model.training.data.iterators.factory;


import ampcontrol.model.training.data.iterators.MiniEpochDataSetIterator;
import ampcontrol.model.training.data.iterators.WorkSpaceWrappingIterator;

/**
 * Wraps {@link MiniEpochDataSetIterator}s from a given factory in a {@link WorkSpaceWrappingIterator}.
 *
 * @param <V> Type of input needed for sourceFactory
 * @author Christian Skärby
 */
public class WorkSpaceWrapping<V> implements DataSetIteratorFactory<MiniEpochDataSetIterator, V> {

    private final DataSetIteratorFactory<MiniEpochDataSetIterator, V> sourceFactory;

    public WorkSpaceWrapping(DataSetIteratorFactory<MiniEpochDataSetIterator, V> sourceFactory) {
        this.sourceFactory = sourceFactory;
    }

    @Override
    public MiniEpochDataSetIterator create(V input) {
        return new WorkSpaceWrappingIterator(sourceFactory.create(input));
    }
}
