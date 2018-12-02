package ampcontrol.model.training.model.evolve.fitness;

import java.util.function.Consumer;
import java.util.function.DoubleUnaryOperator;

/**
 * Transforms fitness score from a source {@link FitnessPolicy} through a {@link DoubleUnaryOperator}
 * @param <T>
 *
 * @author Christian Skärby
 */
public class TransformFitness<T> implements FitnessPolicy<T> {

    private final DoubleUnaryOperator transform;
    private final FitnessPolicy<T> source;

    public TransformFitness(DoubleUnaryOperator transform, FitnessPolicy<T> source) {
        this.transform = transform;
        this.source = source;
    }

    @Override
    public T apply(T candidate, Consumer<Double> fitnessListener) {
        return source.apply(candidate, fitness -> fitnessListener.accept(transform.applyAsDouble(fitness)));
    }
}
