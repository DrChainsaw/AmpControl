package ampcontrol.model.training.model.evolve.fitness;

import org.apache.commons.lang.mutable.MutableDouble;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
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

    /**
     * Test that fitness is calculated correctly
     */
    @Test
    public void applyManyCandidates() {
        final CallableFitnessPolicy<String> callable1 = new CallableFitnessPolicy<>();
        final CallableFitnessPolicy<String> callable2 = new CallableFitnessPolicy<>();
        final FitnessPolicy<String> policy = CombinePolicy.<String>builder()
                .add(callable1)
                .add(callable2)
                .build();

        final MutableDouble fitness1 = new MutableDouble(0);
        final MutableDouble fitness2 = new MutableDouble(0);
        policy.apply("test", fitness1::setValue);
        policy.apply("test", fitness2::setValue);
        assertEquals("No score shall be reported!", 0, fitness1.doubleValue(), 1e-10);
        assertEquals("No score shall be reported!", 0, fitness2.doubleValue(), 1e-10);

        callable1.report(3d);
        assertEquals("No score shall be reported!", 0, fitness1.doubleValue(), 1e-10);
        assertEquals("No score shall be reported!", 0, fitness2.doubleValue(), 1e-10);

        callable1.report(5d);
        assertEquals("No score shall be reported!", 0, fitness1.doubleValue(), 1e-10);
        assertEquals("No score shall be reported!", 0, fitness2.doubleValue(), 1e-10);

        callable2.report(4d);
        assertEquals("Incorrect score!", 12d, fitness1.doubleValue(), 1e-10);
        assertEquals("Incorrect score!", 12d, fitness2.doubleValue(), 1e-10);

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

    private final static class CallableFitnessPolicy<T> implements FitnessPolicy<T> {

        private List<Consumer<Double>> fitnessListeners = new ArrayList<>();

        @Override
        public T apply(T candidate, Consumer<Double> fitnessListener) {
            fitnessListeners.add(fitnessListener);
            return candidate;
        }

        private void report(double fitness) {
            fitnessListeners.forEach(listener -> listener.accept(fitness));
        }
    }
}