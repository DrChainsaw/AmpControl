package ampcontrol.model.training.model.evolve.mutate.layer.blockfunctions;

import ampcontrol.model.training.model.layerblocks.IdBlock;
import ampcontrol.model.training.model.layerblocks.graph.ResBlock;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link ResBlockFunction}
 *
 * @author Christian Skärby
 */
public class ResBlockFunctionTest {

    /**
     * Test that a residual block is created. Trivial test for a trivial class...
     */
    @Test
    public void apply() {
        assertEquals("Incorrect type!", ResBlock.class, new ResBlockFunction(nOut -> new IdBlock()).apply(1L).getClass());
    }
}