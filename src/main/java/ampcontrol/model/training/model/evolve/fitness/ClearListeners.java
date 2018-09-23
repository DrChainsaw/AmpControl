package ampcontrol.model.training.model.evolve.fitness;

import ampcontrol.model.training.model.ModelAdapter;

import java.util.Collections;
import java.util.function.Consumer;

/**
 * {@link FitnessPolicy} which only clears listeners of each {@link ModelAdapter}. Use this if other
 * {@link FitnessPolicy FitnessPolicies} will add listeners to prevent duplicate listeners for unaltered candidates.
 * @param <T>
 */
public class ClearListeners<T extends ModelAdapter>  implements FitnessPolicy<T> {
    @Override
    public T apply(T candidate, Consumer<Double> fitnessListener) {
        candidate.asModel().setListeners(Collections.emptyList());
        return candidate;
    }
}
