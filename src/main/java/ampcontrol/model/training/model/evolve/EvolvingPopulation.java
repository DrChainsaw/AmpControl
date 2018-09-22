package ampcontrol.model.training.model.evolve;

import ampcontrol.model.training.model.evolve.fitness.FitnessPolicy;
import ampcontrol.model.training.model.evolve.selection.Selection;
import org.nd4j.jita.memory.CudaMemoryManager;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Representation of an evolving population of items. Will perform selection based on the given {@link Selection}
 * whenever fitness is obtained for the whole population. What is usually called EvolutionEngine in more
 * sophisticated genetic algorithm frameworks.
 * @param <T>
 *
 * @author Christian Skärby
 */
public final class EvolvingPopulation<T> implements Evolving<EvolvingPopulation<T>>, Population<T> {

    private static final Logger log = LoggerFactory.getLogger(EvolvingPopulation.class);

    private final List<T> population;
    private final List<Map.Entry<Double, T>> evalCands = new ArrayList<>();
    private final FitnessPolicy<T> fitnessPolicy;
    private final Selection<T> selection;
    private final List<Runnable> onChangeCallback = new ArrayList<>();

    public EvolvingPopulation(
            List<T> population,
            FitnessPolicy<T> fitnessPolicy,
            Selection<T> selection) {
        this.population = population.stream()
                .map(cand -> fitnessPolicy.apply(cand, fitness -> reportFitness(fitness, cand)))
                .collect(Collectors.toList());
        this.fitnessPolicy = fitnessPolicy;
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
            log.debug("Callback to: " + onChangeCallback);
            onChangeCallback.forEach(Runnable::run);
        }
    }

    public EvolvingPopulation<T> evolve() {
        if(evalCands.size() != population.size()) {
            throw new IllegalStateException("Must have fitness for all candidates before evolving!");
        }

        log.info("Evolving population...");
        population.clear();

        population.addAll(selection.selectCandiates(evalCands)
                .map(cand -> fitnessPolicy.apply(cand, fitness -> reportFitness(fitness, cand)))
                .collect(Collectors.toList()));
        evalCands.clear();

        // Its either this or catch an exception since everything but the CudaMemoryManager throws an exception
        if(Nd4j.getMemoryManager() instanceof CudaMemoryManager) {
            // Needed to free memory?
            Nd4j.getMemoryManager().purgeCaches();
        }
        return this;
    }

    @Override
    public Stream<T> streamPopulation() {
        if (evalCands.size() == population.size()) {
            evolve();
        }
        return population.stream();
    }

    @Override
    public void onChangeCallback(Runnable callback) {
        onChangeCallback.add(callback);
    }

}
