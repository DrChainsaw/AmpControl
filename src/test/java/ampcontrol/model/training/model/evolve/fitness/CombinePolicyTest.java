package ampcontrol.model.training.model.evolve.fitness;

import org.apache.commons.lang.mutable.MutableDouble;
import org.junit.Test;

import java.util.function.Consumer;
import java.util.stream.Stream;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link CombinePolicy}
 *
 * @author Christian Skärby
 */
public class CombinePolicyTest {

    /**
     * Test that fitness is calculated correctly
     */
    @Test
    public void apply() {
        final FitnessPolicy<String> policy = CombinePolicy.<String>builder()
                .combiner((d1,d2) -> d1 + d2 / 2)
                .add(new FixedFitnessPolicy<>(1d))
                .add(new FixedFitnessPolicy<>(20d,30d))
                .aggregationMethod((d1,d2) -> d1*d2)
                .add(new FixedFitnessPolicy<>(7000d))
                .build();

        final MutableDouble fitness = new MutableDouble(0);
        final String retCand = policy.apply("test", fitness::setValue);
        assertEquals("Candidate shall not change!", "test", retCand);
        assertEquals("Incorrect score!", 1 + (20*30)/2d + 7000/2d, fitness.doubleValue(), 1e-10);

        // Calculate again to verify that things are cleared
        fitness.setValue(0);
        policy.apply("test", fitness::setValue);
        assertEquals("Incorrect score!", 1 + (20*30)/2d + 7000/2d, fitness.doubleValue(), 1e-10);
    }

    private final static class FixedFitnessPolicy<T> implements FitnessPolicy<T> {

        private final Double[] fitness;

        private FixedFitnessPolicy(Double... fitness) {
            this.fitness = fitness;
        }

        @Override
        public T apply(T candidate, Consumer<Double> fitnessListener) {
            Stream.of(fitness).forEach(fitnessListener);
            return candidate;
        }
    }
}