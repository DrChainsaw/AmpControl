package ampcontrol.model.training.model.evolve;

import java.util.function.Function;
import java.util.stream.Stream;

/**
 * Transforms a {@link Population} of type T into a {@link Population} of type V
 * @param <T> Type of source population
 * @param <V> Type of this population
 */
public class TransformPopulation<T,V> implements Population<V> {

    private final Population<T> sourcePopulation;
    private final Function<T,V> transform;

    /**
     * Constructor
     * @param transform Function to transform the population
     * @param sourcePopulation Source population to be transformed
     */
    public TransformPopulation(Function<T, V> transform, Population<T> sourcePopulation) {
        this.sourcePopulation = sourcePopulation;
        this.transform = transform;
    }

    @Override
    public Stream<V> streamPopulation() {
        return sourcePopulation.streamPopulation().map(transform);
    }

    @Override
    public void onChangeCallback(Runnable callback) {
        // Stateless (assumes transform is stateless too).
        sourcePopulation.onChangeCallback(callback);
    }
}
