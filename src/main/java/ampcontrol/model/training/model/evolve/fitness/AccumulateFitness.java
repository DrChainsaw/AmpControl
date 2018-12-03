package ampcontrol.model.training.model.evolve.fitness;

import java.util.function.Consumer;
import java.util.function.DoubleBinaryOperator;

/**
 * Accumulates a predetermined number fitness score samples before giving to the provided consumer
 *
 * @author Christian Skärby
 */
public class AccumulateFitness<T> implements FitnessPolicy<T> {

    private final int nrofSamplesToAccumulate;
    private final DoubleBinaryOperator accumulation;
    private final double identity;
    private final FitnessPolicy<T> source;


    public AccumulateFitness(
            int nrofSamplesToAccumulate,
            DoubleBinaryOperator accumulation,
            double identity,
            FitnessPolicy<T> source) {
        this.nrofSamplesToAccumulate = nrofSamplesToAccumulate;
        this.accumulation = accumulation;
        this.identity = identity;
        this.source = source;
    }

    @Override
    public T apply(T candidate, Consumer<Double> fitnessListener) {

        final Consumer<Double> fitnessAccumulator = new Consumer<Double>() {
            int cnt = 0;
            double scoreAcc = identity;

            @Override
            public void accept(Double score) {
                scoreAcc = accumulation.applyAsDouble(scoreAcc, score);
                cnt++;
                if (cnt == nrofSamplesToAccumulate) {
                    fitnessListener.accept(scoreAcc);
                    cnt = 0;
                }
            }
        };


        return source.apply(candidate, fitnessAccumulator);
    }
}
