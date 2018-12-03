package ampcontrol.model.training.model.evolve.fitness;

import ampcontrol.model.training.model.ModelAdapter;

import java.util.function.Consumer;

public class NumberOfParametersPolicy<T extends ModelAdapter> implements FitnessPolicy<T> {

    @Override
    public T apply(T candidate, Consumer<Double> fitnessListener) {
        fitnessListener.accept((double)candidate.asModel().numParams());
        return candidate;
    }
}
