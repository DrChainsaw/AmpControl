package ampcontrol.model.training.model.evolve.fitness;

import java.util.function.Consumer;

/**
 * A policy for measuring fitness of candidates.
 *
 * @param <T>
 * @author Christian Skärby
 */
public interface FitnessPolicy<T> {

    /**
     * Apply the scoring policy to the given candidate
     * @param candidate Candidate for which fitness score shall be measured
     * @param fitnessListener Listens to fitness score
     * @return The candidate for which the fitness policy has been applied
     */
    T apply(T candidate, Consumer<Double> fitnessListener);

}
