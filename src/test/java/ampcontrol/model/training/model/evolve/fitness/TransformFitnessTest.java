package ampcontrol.model.training.model.evolve.fitness;

import org.apache.commons.lang.mutable.MutableDouble;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link TransformFitness}
 *
 * @author Christian Skärby
 */
public class TransformFitnessTest {

    /**
     * Test that transform is applied
     */
    @Test
    public void apply() {
        MutableDouble actualFitness = new MutableDouble(0);
        Double retCand = new TransformFitness<Double>(d -> d*13, (cand, fl) -> {
            fl.accept(cand);
            return cand;
        }).apply(27d, actualFitness::setValue);
        assertEquals("Incorrect candidate returned!", 27d, retCand, 1e-10);
        assertEquals("Incorrect fitness!", 13*27d, actualFitness.doubleValue(), 1e-10);
    }
}