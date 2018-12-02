package ampcontrol.model.training.model.evolve.fitness;

import org.apache.commons.lang.mutable.MutableDouble;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link AccumulateFitness}
 *
 * @author Christian Skärby
 */
public class AccumulateFitnessTest {

    /**
     * Test that fitness is accumulated
     */
    @Test
    public void apply() {
        final FitnessPolicy<Double> policy = new AccumulateFitness<>(3, (d1,d2) -> d1*d2,1, (cand, listener) -> {
            listener.accept(cand);
            listener.accept(cand+1);
            listener.accept(cand+2);
            return cand;
        });

        final MutableDouble actualFitness = new MutableDouble(-1);
        assertEquals("Incorrect candidate returned!", 13d, policy.apply(13d, actualFitness::setValue), 1e-10);
        assertEquals("Incorrect fitness!", 13*14*15d, actualFitness.doubleValue(), 1e-10);
    }
}