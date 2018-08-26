package ampcontrol.model.training.data.iterators.factory;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * Interface for creating {@link DataSetIterator}s from input.
 * TODO: Design needs rework. Generic input is not good for substitution!
 *
 * @param <T> Type of {@link DataSetIterator} created
 * @param <V> Type of input needed
 * @author Christian Skärby
 */
public interface DataSetIteratorFactory<T extends DataSetIterator, V> {

    /**
     * Create a {@link T} from the given input {@link V}
     * @param input Needed input data
     * @return a new {@link T}
     */
    T create(V input);

}
