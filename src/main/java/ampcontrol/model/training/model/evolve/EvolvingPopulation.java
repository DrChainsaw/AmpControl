package ampcontrol.model.training.model.evolve;

import ampcontrol.model.training.model.evolve.selection.Selection;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.function.BiConsumer;
import java.util.stream.Collectors;

/**
 * Representation of a population of {@link Evolving} items. Will perform selection based on the given {@link Selection}
 * whenever fitness is obtained for the whole population.
 * @param <T>
 *
 * @author Christian Skärby
 */
public final class EvolvingPopulation<T> implements Evolving<EvolvingPopulation<T>> {

    private static final Logger log = LoggerFactory.getLogger(EvolvingPopulation.class);

    private final List<T> population;
    private final List<Map.Entry<Double, T>> evalCands = new ArrayList<>();
    private final EvolutionCallback<T> callback;
    private final Selection<T> selection;

    public interface EvolutionCallback<T> extends BiConsumer<List<T>, BiConsumer<Double, T>> {/* Just a an alias */
    }

    public EvolvingPopulation(
            List<T> population,
            EvolutionCallback<T> callback,
            Selection<T> selection) {
        this.population = population;
        this.callback = callback;
        this.selection = selection;
    }

    private void reportFitness(double fitness, T item) {
        if(!population.contains(item)) {
            throw new IllegalArgumentException("Got item which is not part of population: " + item + "! Population: " + population);
        }

        if(evalCands.stream().map(Map.Entry::getValue).anyMatch(item::equals)) {
            throw new IllegalArgumentException("Got duplicate item: " + item + "! evalCands: " + evalCands);
        }

        log.info("got fitness " + fitness + " for cand " + population.indexOf(item) + " size: " + evalCands.size());
        evalCands.add(new AbstractMap.SimpleEntry<>(fitness, item));

        if (evalCands.size() == population.size()) {
            evolve();
        }
    }

    public EvolvingPopulation<T> evolve() {
        if(evalCands.size() != population.size()) {
            throw new IllegalStateException("Must have fitness for all candidates before evolving!");
        }

        log.info("Evolving population...");
        population.clear();

        population.addAll(selection.selectCandiates(evalCands)
                .collect(Collectors.toList()));
        evalCands.clear();
        initEvolvingPopulation();
        // Needed to get free memory
        Nd4j.getMemoryManager().purgeCaches();
        return this;
    }

    public void initEvolvingPopulation() {
        callback.accept(Collections.unmodifiableList(population), this::reportFitness);
    }
}
