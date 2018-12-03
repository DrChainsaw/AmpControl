package ampcontrol.model.training.model.evolve;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * A cached population. Useful when source {@link Population} does significant things before streaming the population.
 * Note that cache must be cleared manually
 * @param <T>
 * @author Christian Skärby
 */
public class CachedPopulation<T> implements Population<T> {
    private static final Logger log = LoggerFactory.getLogger(CachedPopulation.class);

    private final Population<T> sourcePopulation;
    private Collection<T> cachedPopulation;
    private final List<Runnable> onChangeCallback = new ArrayList<>();

    public CachedPopulation(Population<T> sourcePopulation) {
        this.sourcePopulation = sourcePopulation;
        sourcePopulation.onChangeCallback(() -> cachedPopulation = null);
    }

    @Override
    public Stream<T> streamPopulation() {
        if(cachedPopulation == null) {
            log.info("Refreshing cache");
            cachedPopulation = sourcePopulation.streamPopulation().collect(Collectors.toList());
            onChangeCallback.forEach(Runnable::run);
        }
        return cachedPopulation.stream();
    }

    @Override
    public void onChangeCallback(Runnable callback) {
        onChangeCallback.add(callback);
    }
}
