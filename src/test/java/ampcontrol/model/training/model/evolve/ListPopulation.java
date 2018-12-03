package ampcontrol.model.training.model.evolve;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

/**
 * Simple population backed by a list. For testing purposes
 *
 * @param <T>
 * @author Christian Skärby
 */
public class ListPopulation<T> implements Population<T> {

    private final List<T> population;
    private final List<Runnable> callbacks = new ArrayList<>();

    public ListPopulation(List<T> population) {
        this.population = population;
    }

    @Override
    public Stream<T> streamPopulation() {
        return population.stream();
    }

    @Override
    public void onChangeCallback(Runnable callback) {
        callbacks.add(callback);
    }

    List<Runnable> getCallbacks() {return callbacks;}
}
